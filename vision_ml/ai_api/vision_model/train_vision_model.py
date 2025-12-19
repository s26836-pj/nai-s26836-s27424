import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    balanced_accuracy_score
)

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import ConfusionMatrixDisplay

"""
Projekt: Klasyfikacja stanu roślin (Fittonia) na podstawie obrazów
         z wykorzystaniem sieci konwolucyjnych (CNN)

Opis problemu:
Program realizuje klasyfikację obrazów roślin Fittonia na podstawie zdjęć
liści. Każdy obraz przypisany jest do jednej z klas opisujących stan rośliny
(np. low, mid, perfect).

Zastosowane metody i techniki:
- Transfer learning z wykorzystaniem MobileNetV2 (ImageNet)
- Zamrożony backbone + własna głowa klasyfikacyjna
- Augmentacja danych obrazowych
- Categorical Focal Loss (redukcja wpływu łatwych przykładów)
- 5-fold Stratified Cross-Validation
- Metryki: Accuracy, Macro F1, Balanced Accuracy
- Macierze pomyłek zapisywane jako pliki PNG

Pipeline obejmuje:
- Wczytanie adnotacji i weryfikację istnienia plików obrazów
- Mapowanie klas tekstowych na indeksy liczbowe
- Budowę datasetów tf.data dla treningu, walidacji i ewaluacji
- Trening i walidację modelu w schemacie cross-validation
- Trening modelu końcowego na pełnym zbiorze danych
- Zapis modelu, metadanych eksperymentu oraz mapowania klas

Autorzy:
    Błażej Kanczkowski (s26836)
    Adam Rzepa (s27424)

Instrukcja uruchomienia:
    README.md
"""
tf.keras.utils.set_random_seed(42)
BASE_DIR = Path(__file__).resolve().parent

# target_state jest PROBLEMEM ORDINALNYM (low < mid < perfect),
# a uczymy klasyfikacji kategorycznej (softmax).
# Pomyłki low<->mid są karane tak samo jak low<->perfect.
# Docelowo: ordinal regression (CORAL / cumulative logits).

def main():
    """
       Główna funkcja treningowa dla klasyfikacji obrazów Fittonia.

       Pipeline:
       - wczytanie adnotacji i sprawdzenie istnienia plików
       - mapowanie klas tekstowych na indeksy
       - przygotowanie datasetów tf.data (augmentacja + preprocessing)
       - 5-fold Stratified Cross-Validation
       - trening MobileNetV2 (transfer learning)
       - ewaluacja (accuracy, macro F1, confusion matrix)
       - trening modelu końcowego na pełnym zbiorze
       - zapis modelu, metadanych i mapowania klas
       """
    OUTPUT_DIR = BASE_DIR / "vision_data"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    IMAGES_DIR = (BASE_DIR / "../../images").resolve()
    ANNOTATIONS_PATH = (BASE_DIR / "../../csv/annotations.csv").resolve()

    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    AUTOTUNE = tf.data.AUTOTUNE # wątki mapowanie etc.

    df = pd.read_csv(ANNOTATIONS_PATH)
    df['target_state'] = df['target_state'].astype(str)

    df['exists'] = df['filename'].apply(
        lambda f: (IMAGES_DIR / f).exists()
    )

    missing = df[~df['exists']]
    if not missing.empty:
        print("UWAGA! Brakujące pliki:\n", missing[['filename', 'target_state']])
    df = df[df['exists']].drop(columns=['exists'])

    num_classes = df['target_state'].nunique()
    print("Liczba klas:", num_classes, df['target_state'].unique())
    print("Łącznie próbek:", len(df))

    classes = sorted(df["target_state"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    df["class_idx"] = df["target_state"].map(class_to_idx).astype(int)

    def categorical_focal_loss(gamma=2.0):
        """
            Implementacja categorical focal loss.

            Funkcja zmniejsza wagę łatwych przykładów i skupia uczenie
            na próbkach trudnych do sklasyfikowania.

            Args:
                gamma (float): Parametr regulujący siłę skupienia na trudnych próbkach.

            Returns:
                function: Funkcja straty kompatybilna z Keras.
            """
        cce = tf.keras.losses.CategoricalCrossentropy(
            from_logits=False,
            reduction=tf.keras.losses.Reduction.NONE
        )

        def loss(y_true, y_pred):
            ce = cce(y_true, y_pred)
            p_true = tf.reduce_sum(y_true * y_pred, axis=-1)
            weight = tf.pow(1.0 - p_true, gamma)
            return weight * ce

        return loss

    def load_and_preprocess(path, label_idx):
        """
        Wczytuje obraz z dysku i wykonuje podstawowe przetwarzanie.

        Operacje:
        - odczyt JPEG
        - dekodowanie do RGB
        - resize do IMAGE_SIZE
        - konwersja do float32

        Args:
            path (tf.Tensor): Ścieżka do pliku obrazu.
            label_idx (tf.Tensor): Indeks klasy.

        Returns:
            tuple: (obraz, indeks klasy)
        """
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, IMAGE_SIZE)
        image = tf.cast(image, tf.float32)
        return image, label_idx

    def augment(image, label_idx):
        """
           Wykonuje augmentację danych obrazowych.

           Operacje:
           - losowe odbicie poziome
           - losowy centralny crop
           - losowa zmiana jasności

           Args:
               image (tf.Tensor): Obraz wejściowy.
               label_idx (tf.Tensor): Indeks klasy.

           Returns:
               tuple: (zaaugmentowany obraz, indeks klasy)
           """
        image = tf.image.random_flip_left_right(image)

        image = tf.image.central_crop(
            image,
            central_fraction=tf.random.uniform([], 0.85, 1.0)
        )
        image = tf.image.resize(image, IMAGE_SIZE)

        image = tf.image.random_brightness(
            image,
            max_delta=0.2 * 255.0
        )

        return image, label_idx

    def apply_preprocess_and_one_hot(image, label_idx):
        """
           Stosuje preprocessing MobileNetV2 oraz kodowanie one-hot etykiety.

           Args:
               image (tf.Tensor): Obraz po augmentacji.
               label_idx (tf.Tensor): Indeks klasy.

           Returns:
               tuple:
                   - obraz po preprocess_input
                   - etykieta one-hot
           """
        image = preprocess_input(image) # dostosowanie RGB do Mobilenet
        label = tf.one_hot(label_idx, depth=num_classes)
        return image, label

    def make_dataset(paths, label_indices, augment_data, shuffle_data=True):
        """
            Buduje dataset tf.data dla treningu lub walidacji.

            Args:
                paths (array-like): Ścieżki do obrazów.
                label_indices (array-like): Indeksy klas.
                augment_data (bool): Czy stosować augmentację.
                shuffle_data (bool): Czy tasować dane.

            Returns:
                tf.data.Dataset: Gotowy dataset do treningu lub walidacji.
            """
        ds = tf.data.Dataset.from_tensor_slices((paths, label_indices))

        if shuffle_data:
            ds = ds.shuffle(
                buffer_size=len(paths),
                reshuffle_each_iteration=True
            )

        ds = ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)

        if augment_data:
            ds = ds.map(augment, num_parallel_calls=AUTOTUNE)

        ds = ds.map(apply_preprocess_and_one_hot, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
        return ds

    def make_eval_dataset(paths):
        """
         Tworzy dataset do ewaluacji lub predykcji (bez etykiet).

         Args:
             paths (array-like): Ścieżki do obrazów.

         Returns:
             tf.data.Dataset: Dataset obrazów po preprocess_input.
         """
        ds = tf.data.Dataset.from_tensor_slices(paths)

        def _load(path):
            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, IMAGE_SIZE)
            image = tf.cast(image, tf.float32)
            return preprocess_input(image)

        ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
        return ds

    def build_model(num_classes):
        """
          Buduje i kompiluje model CNN oparty o MobileNetV2.

          Architektura:
          - MobileNetV2 (zamrożony backbone)
          - GlobalAveragePooling
          - Dense(128) + L2
          - Dropout
          - Softmax

          Args:
              num_classes (int): Liczba klas wyjściowych.

          Returns:
              tf.keras.Model: Skompilowany model Keras.
          """
        base_model = MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights="imagenet" #cechy imagenet
        )
        base_model.trainable = False

        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(
                128,
                activation="relu",
                kernel_regularizer=regularizers.l2(1e-4) #kara za zbyt duza wage dla cechy
            ),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation="softmax")
        ])

        model.compile(
            optimizer=Adam(1e-4),
            loss=categorical_focal_loss(gamma=2.0),
            metrics=["accuracy"]
        )
        return model

    def save_confusion_matrix_png(cm, classes, title, out_path):
        """
            Zapisuje macierz pomyłek jako plik PNG.

            Args:
                cm (np.ndarray): Macierz pomyłek.
                classes (list): Nazwy klas.
                title (str): Tytuł wykresu.
                out_path (Path): Ścieżka zapisu pliku PNG.
            """
        fig, ax = plt.subplots(figsize=(6, 6))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=classes
        )
        disp.plot(ax=ax, cmap="Greens", values_format="d", colorbar=False)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close(fig)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    X = df["filename"].values
    y_idx = df["class_idx"].values

    fold = 1
    fold_val_acc = []
    fold_val_loss = []
    fold_confusion_matrices = []
    fold_val_f1 = []

    for train_idx, val_idx in skf.split(X, y_idx):
        print(f"\n\n============FOLD {fold}\n")

        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        train_paths = train_df["filename"].apply(lambda f: str(IMAGES_DIR / f)).values

        val_paths = val_df["filename"].apply(lambda f: str(IMAGES_DIR / f)).values

        train_ds = make_dataset(train_paths, train_df["class_idx"].values, True, shuffle_data=True)
        val_ds = make_dataset(val_paths, val_df["class_idx"].values, False, shuffle_data=False)

        model = build_model(num_classes)

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            ModelCheckpoint(OUTPUT_DIR / f"fittonia_fold_{fold}.keras", save_best_only=True),
            CSVLogger(OUTPUT_DIR / f"log_fold_{fold}.csv")

        ]

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=100,
            callbacks=callbacks,
            verbose=1
            # class_weight=class_weight
        )
        val_eval_ds = make_eval_dataset(val_paths)

        y_val_true = val_df["class_idx"].values
        y_val_pred = np.argmax(
            model.predict(val_eval_ds, verbose=0),
            axis=1
        )

        cm_fold = confusion_matrix(y_val_true, y_val_pred)
        f1_macro = f1_score(y_val_true, y_val_pred, average="macro")
        fold_confusion_matrices.append(cm_fold)
        fold_val_f1.append(f1_macro)

        png_path = OUTPUT_DIR / f"confusion_matrix_fold_{fold}.png"
        save_confusion_matrix_png(
            cm=cm_fold,
            classes=classes,
            title=f"Confusion Matrix – Fold {fold}",
            out_path=png_path
        )

        val_loss, val_acc = model.evaluate(val_ds, verbose=0)
        fold_val_loss.append(val_loss)
        fold_val_acc.append(val_acc)

        fold += 1

    print("\nCV mean acc:", np.mean(fold_val_acc))
    print("CV std acc:", np.std(fold_val_acc))

    print("CV mean macro F1:", np.mean(fold_val_f1))
    print("CV std macro F1:", np.std(fold_val_f1))

    print("\nTrening modelu końcowego...")

    full_paths = df["filename"].apply(lambda f: str(IMAGES_DIR / f)).values
    full_ds = make_dataset(full_paths, df["class_idx"].values, True)

    final_model = build_model(num_classes)

    history_final = final_model.fit(
        full_ds,
        epochs=100,
        callbacks=[
            EarlyStopping(monitor="loss", patience=3, restore_best_weights=True),
            ModelCheckpoint(OUTPUT_DIR / "fittonia_final_model.keras", save_best_only=True)
        ],
        verbose=1
    )

    class_indices = {
        "class_to_idx": class_to_idx,
        "idx_to_class": {str(k): v for k, v in idx_to_class.items()}
    }

    with open(OUTPUT_DIR / "class_indices.json", "w", encoding="utf-8") as f:
        json.dump(class_indices, f, ensure_ascii=False, indent=2)

    eval_ds = make_eval_dataset(full_paths)
    y_true = df["class_idx"].values
    y_pred = np.argmax(final_model.predict(eval_ds), axis=1)

    cm = confusion_matrix(y_true, y_pred)
    save_confusion_matrix_png(
        cm=cm,
        classes=classes,
        title="Confusion Matrix – Final Model (TRAIN SET)",
        out_path=OUTPUT_DIR / "confusion_matrix_final_model.png"
    )

    print(classification_report(
        y_true,
        y_pred,
        target_names=[idx_to_class[i] for i in range(num_classes)]
    ))

    print("Balanced accuracy:",
          balanced_accuracy_score(y_true, y_pred))

    metadata = {
        "num_classes": num_classes,
        "classes": classes,
        "num_samples": len(df),
        "cv_folds": 5,
        "cv_mean_accuracy": float(np.mean(fold_val_acc)),
        "cv_std_accuracy": float(np.std(fold_val_acc)),
        "cv_mean_macro_f1": float(np.mean(fold_val_f1)),
        "cv_std_macro_f1": float(np.std(fold_val_f1)),
        "loss": "categorical_focal_loss(gamma=2.0)",
        "architecture": "MobileNetV2 + GAP + Dense(128)",
        "note": "Final model evaluated on TRAIN SET only",
    }

    with open(OUTPUT_DIR / "experiment_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    # UWAGA:
    # Ewaluacja wykonana na TRAIN SET.
    # Do realnej jakości potrzebny HOLD-OUT TEST SET.
if __name__ == "__main__":
    main()