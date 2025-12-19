import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    f1_score,
    multilabel_confusion_matrix
)

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping

"""
Projekt: Inteligentny system monitorowania stanu roślin (Fittonia)
         – model decyzyjny oparty na danych sensorowych

Opis:
Moduł implementuje DRUGI model sztucznej inteligencji w systemie,
który współpracuje z modelem wizyjnym (CNN – MobileNetV2).

Model ten analizuje dane z czujników środowiskowych oraz cechy czasowe
i przewiduje zestaw binarnych decyzji opisujących aktualne potrzeby rośliny.

Predykcje modelu sensorowego:
- forgot_to_water      – czy roślina wymaga podlania
- too_dark_today       – czy poziom światła jest niewystarczający
- worth_relocating     – czy warto zmienić lokalizację rośliny

Integracja z modelem wizyjnym:
Model sensorowy stanowi uzupełnienie modelu CV, który klasyfikuje wizualny
stan rośliny (np. low / mid / perfect). Wspólnie modele tworzą
dwupoziomowy system decyzyjny:
    1) Model wizyjny – ocena kondycji na podstawie obrazu
    2) Model sensorowy – rekomendacje działań na podstawie danych liczbowych

Zastosowane metody:
- Wieloetykietowa klasyfikacja binarna (multi-label)
- Sieć MLP (Dense + BatchNorm + Dropout)
- Funkcja straty: binary cross-entropy
- Standaryzacja cech (StandardScaler)
- 5-fold Stratified Cross-Validation (stratyfikacja po kombinacji etykiet)
- Metryki: Macro F1, Weighted F1, Micro F1
- Macierze pomyłek dla każdej etykiety (PNG + CSV)

Pipeline obejmuje:
- Wczytanie danych po przetworzeniu z czujników
- Skalowanie cech wejściowych
- Cross-validation modelu
- Trening modelu końcowego na pełnym zbiorze danych
- Zapis modelu, skalera, etykiet oraz pełnych metryk eksperymentu

Autorzy:
    Błażej Kanczkowski (s26836)
    Adam Rzepa (s27424)

Instrukcja uruchomienia:
    README.md
"""

BASE_DIR = Path(__file__).resolve().parent
tf.keras.utils.set_random_seed(42)

def main():
    """
    Główna funkcja treningowa modelu sensorowego (drugi model AI).

    Model realizuje wieloetykietową klasyfikację binarną na podstawie
    danych z czujników środowiskowych oraz cech czasowych.

    Etapy:
    - wczytanie i przygotowanie danych sensorowych
    - standaryzacja cech wejściowych
    - 5-fold Stratified Cross-Validation (stratyfikacja po kombinacjach etykiet)
    - ewaluacja przy użyciu Macro F1 i val_loss
    - trening modelu końcowego na pełnym zbiorze
    - zapis modelu, skalera, etykiet i metryk

    Model ten współpracuje z modelem wizyjnym (CNN),
    tworząc dwupoziomowy system decyzyjny dla roślin.
    """

    OUTPUT_DIR = BASE_DIR / "sensor_data"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    DATA_PATH = BASE_DIR / "csv" / "Dane_po_przetworzeniu.csv"
    df = pd.read_csv(DATA_PATH, sep=";")

    label_cols = [
        "forgot_to_water",
        "too_dark_today",
        "worth_relocating",
    ]

    feature_cols = [
        "light",
        "temperature",
        "soil_capacitance",
        "hours_since_last_watering",
        "delta_soil",
        "rolling_avg_light_6h",
        "light_hours_today",
        "hour_of_day",
    ]

    X_raw = df[feature_cols].values.astype(np.float32)
    y_raw = df[label_cols].values.astype(np.int32)

    print(f"X_raw shape: {X_raw.shape}")
    print(f"y_raw shape: {y_raw.shape}")

    def build_model(output_dim: int, input_dim: int) -> Model:
        """
          Buduje i kompiluje sieć MLP dla danych sensorowych.

          Architektura:
          - Dense + LeakyReLU
          - Batch Normalization
          - Dropout
          - Warstwa wyjściowa sigmoid (multi-label)

          Args:
              output_dim (int): Liczba etykiet wyjściowych.
              input_dim (int): Liczba cech wejściowych.

          Returns:
              tensorflow.keras.Model: Skompilowany model MLP.
          """
        inp = Input(shape=(input_dim,))
        x = Dense(128)(inp)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)

        x = Dense(64)(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        x = Dense(32)(x)
        x = LeakyReLU()(x)

        out = Dense(output_dim, activation="sigmoid")(x)
        model = Model(inputs=inp, outputs=out)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss="binary_crossentropy"
        )
        return model


    print("\n5-fold cross-validation")

    combo_labels = np.array(
        ["".join(map(str, row.astype(int))) for row in y_raw]
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold = 1
    fold_val_loss = []
    fold_val_f1_macro = []
    fold_summaries = []

    # 5-fold cross-validation
    # Trening i walidacja modelu sensorowego w schemacie
    # StratifiedKFold po kombinacji etykiet
    for train_idx, val_idx in skf.split(X_raw, combo_labels):
        print("\n" + "=" * 30)
        print(f"FOLD {fold}")
        print("=" * 30)

        X_tr_raw, X_val_raw = X_raw[train_idx], X_raw[val_idx]
        y_tr, y_val = y_raw[train_idx], y_raw[val_idx]

        scaler_fold = StandardScaler()
        X_tr = scaler_fold.fit_transform(X_tr_raw)
        X_val = scaler_fold.transform(X_val_raw)

        model_cv = build_model(output_dim=len(label_cols),
                                    input_dim=X_tr.shape[1])

        cb = [
            EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True
            )
        ]

        history = model_cv.fit(
            X_tr,
            y_tr,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=cb,
            verbose=1,
        )

        hist_df = pd.DataFrame(history.history)
        hist_df.to_csv(
            OUTPUT_DIR / f"training_log_fold_{fold}.csv",
            index=False,
        )

        val_losses = history.history["val_loss"]
        best_epoch = int(np.argmin(val_losses))
        best_loss = float(val_losses[best_epoch])

        y_val_pred = (model_cv.predict(X_val) > 0.5).astype(int)
        f1_macro = f1_score(y_val, y_val_pred, average="macro")
        fold_val_loss.append(best_loss)
        fold_val_f1_macro.append(f1_macro)

        print(
            f"FOLD {fold} - best val_loss: {best_loss:.4f}, "
            f"F1 macro (val): {f1_macro:.4f}"
        )

        fold_summaries.append(
            {
                "fold": fold,
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "best_epoch": best_epoch + 1,
                "best_val_loss": best_loss,
                "val_f1_macro": float(f1_macro),
            }
        )

        fold += 1

    print("\nPODSUMOWANIE 5-FOLD CV")
    print("Val F1 macro per fold:", [f"{f:.4f}" for f in fold_val_f1_macro])
    print(
        f"Średni F1 macro: {np.mean(fold_val_f1_macro):.4f} "
        f"+/- {np.std(fold_val_f1_macro):.4f}"
    )
    print("Val loss per fold:", [f"{l:.4f}" for l in fold_val_loss])
    print(
        f"Średnia val_loss: {np.mean(fold_val_loss):.4f} "
        f"+/- {np.std(fold_val_loss):.4f}"
    )

    cv_metrics = {
        "folds": fold_summaries,
        "mean_val_f1_macro": float(np.mean(fold_val_f1_macro)),
        "std_val_f1_macro": float(np.std(fold_val_f1_macro)),
        "mean_val_loss": float(np.mean(fold_val_loss)),
        "std_val_loss": float(np.std(fold_val_loss)),
    }

    print("\n=== Trening modelu końcowego na CAŁYM zbiorze ===")

    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(X_raw)

    final_model = build_model(
        output_dim=len(label_cols),
        input_dim=X_scaled.shape[1],
    )

    cb_final = [
        EarlyStopping(
            monitor="loss",
            patience=10,
            restore_best_weights=True
        )
    ]

    history_final = final_model.fit(
        X_scaled,
        y_raw,
        epochs=100,
        batch_size=32,
        callbacks=cb_final,
        verbose=1,
    )

    hist_final_df = pd.DataFrame(history_final.history)
    hist_final_df.to_csv(
        OUTPUT_DIR / "training_log_final.csv",
        index=False,
    )

    y_pred_proba = final_model.predict(X_scaled)
    y_pred = (y_pred_proba > 0.5).astype(int)

    clf_report = classification_report(
        y_raw,
        y_pred,
        target_names=label_cols,
        output_dict=True,
        zero_division=0,
    )
    print("\nClassification report (full data):")
    print(classification_report(y_raw, y_pred, target_names=label_cols))

    report_df = pd.DataFrame(clf_report).transpose()
    report_df.to_csv(
        OUTPUT_DIR / "classification_report_sensor.csv",
        sep=";",
    )

    # F1
    f1_macro = clf_report["macro avg"]["f1-score"]
    f1_weighted = clf_report["weighted avg"]["f1-score"]
    f1_micro = clf_report["micro avg"]["f1-score"]

    print(f"\nF1 macro (full): {f1_macro}")
    print(f"F1 weighted (full): {f1_weighted}")
    print(f"F1 micro (full): {f1_micro}")

    ml_cm = multilabel_confusion_matrix(y_raw, y_pred)

    cm_rows = []
    for label, cm in zip(label_cols, ml_cm):
        tn, fp, fn, tp = cm.ravel()
        cm_rows.append(
            {"label": label, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
        )
    cm_df = pd.DataFrame(cm_rows)
    cm_df.to_csv(
        OUTPUT_DIR / "multilabel_confusion_matrix.csv",
        index=False,
        sep=";",
    )

    def plot_cm_single(cm, classes, title, out_path, normalize=False):
        """
        Rysuje i zapisuje macierz pomyłek dla pojedynczej etykiety.

        Obsługuje wersję:
        - surową (liczności)
        - znormalizowaną (udziały procentowe)

        Args:
            cm (np.ndarray): Macierz pomyłek 2x2.
            classes (list): Nazwy klas (np. ["neg", "pos"]).
            title (str): Tytuł wykresu.
            out_path (Path): Ścieżka zapisu pliku PNG.
            normalize (bool): Czy normalizować wartości.
        """
        if normalize:
            cm_display = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        else:
            cm_display = cm

        fig, ax = plt.subplots(figsize=(4, 3))
        im = ax.imshow(cm_display, interpolation="nearest", cmap="Blues")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)

        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes)

        fmt = ".2f" if normalize else "d"
        thresh = cm_display.max() / 2.0
        for i in range(cm_display.shape[0]):
            for j in range(cm_display.shape[1]):
                value = cm_display[i, j]
                ax.text(
                    j,
                    i,
                    format(value, fmt),
                    ha="center",
                    va="center",
                    color="white" if value > thresh else "black",
                )

        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)


    for label, cm in zip(label_cols, ml_cm):
        plot_cm_single(
            cm,
            classes=["neg", "pos"],
            title=label,
            out_path=OUTPUT_DIR / f"cm_{label}.png",
            normalize=False,
        )
        plot_cm_single(
            cm,
            classes=["neg", "pos"],
            title=f"{label} (normalized)",
            out_path=OUTPUT_DIR / f"cm_{label}_normalized.png",
            normalize=True,
        )

    model_path = OUTPUT_DIR / "sensor_multilabel_model.keras"
    final_model.save(model_path)

    scaler_path = OUTPUT_DIR / "sensor_scaler.save"

    joblib.dump(scaler_final, scaler_path)

    labels_path = OUTPUT_DIR / "sensor_labels.txt"
    with open(labels_path, "w", encoding="utf-8") as f:
        for lbl in label_cols:
            f.write(lbl + "\n")

    final_history_json = {
        k: [float(x) for x in v] for k, v in history_final.history.items()
    }

    metrics_all = {
        "data_path": str(DATA_PATH),
        "features": feature_cols,
        "labels": label_cols,
        "cv_metrics": cv_metrics,
        "history": final_history_json,
        "classification_report": clf_report,
        "f1_scores": {
            "macro": float(f1_macro),
            "weighted": float(f1_weighted),
            "micro": float(f1_micro),
        },
        "multilabel_confusion_matrix_shape": list(ml_cm.shape),
    }

    metrics_path = OUTPUT_DIR / "metrics_sensor.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_all, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()