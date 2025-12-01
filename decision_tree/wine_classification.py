"""
Projekt: Klasyfikacja jakości wina (Wine Quality)
         przy użyciu Decision Tree i Support Vector Machine (SVM)

Opis problemu:
Program analizuje zbiór danych winequality-white.csv (UCI ML Repository),
a następnie wykonuje klasyfikację binarną jakości wina:
    1 – dobre wino     (quality >= 6)
    0 – słabsze wino   (quality < 6)

W projekcie trenowane są dwa modele:
    - Drzewo decyzyjne (DecisionTreeClassifier)
    - Maszyna wektorów nośnych (SVM) z jądrem RBF

Zawiera również:
    - Wizualizację zależności alcohol vs residual sugar (scatter plot)
    - Heatmap korelacji cech fizykochemicznych
    - Predykcję dla ręcznie zdefiniowanej próbki (w dokładnym formacie CSV)
    - Obliczenie metryk klasyfikacji (Accuracy, Precision, Recall, F1)

Autorzy:
    Błażej Kanczkowski (s26836)
    Adam Rzepa (s27424)

Instrukcja uruchomienia:
README.md
"""
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from utils.ml_utils import (
    split_and_scale,
    train_decision_tree,
    train_svm_with_gridsearch_all_kernels,
    train_svm_rbf,
    evaluate_model,
    predict_example,
)

def load_winne_data(path: Path, threshold=6):
    """
      Wczytuje dane z winequality-white.csv, dodaje kolumnę 'label'
      i zwraca:
          df  - pełny DataFrame (z quality i label)
          X   - cechy (bez quality i label)
          y   - etykieta binarna (0/1)
      threshold:
          minimalna jakość, od której uznajemy wino za "dobre" (label=1)
      """
    df = pd.read_csv(path, sep=";")
    df["label"] = (df["quality"] >= threshold).astype(int)
    X = df.drop(columns=["quality", "label"])
    y = df["label"]

    return df, X, y


def plot_feature_scatter(df: pd.DataFrame):
    """
    Prosta wizualizacja:
    - oś X: alcohol
    - oś Y: residual sugar
    - kolor: label (0 = gorsze, 1 = lepsze wino)
    """
    plt.figure(figsize=(8, 6))

    good = df[df["label"] == 1]
    bad = df[df["label"] == 0]

    plt.scatter(
        bad["alcohol"],
        bad["residual sugar"],
        alpha=0.5,
        label="label=0 (gorsze)",
        marker="o"
    )
    plt.scatter(
        good["alcohol"],
        good["residual sugar"],
        alpha=0.5,
        label="label=1 (lepsze)",
        marker="x"
    )

    plt.xlabel("Alcohol")
    plt.ylabel("Residual sugar")
    plt.title("White wine quality – alcohol vs residual sugar")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("wine_alcohol_sugar_scatter.png")

def predict_custom_wine_sample(dt_model, svm_model, scaler):
    """
    Przykładowa predykcja dla jednej, ręcznie zdefiniowanej próbki wina.
    Dane wejściowe są w tym samym formacie jak w pliku winequality-white.csv:
    'fixed acidity'; 'volatile acidity'; 'citric acid'; ... ; 'alcohol'; 'quality'

    Tutaj quality traktujemy jako "prawdziwą" ocenę jakości, a modele
    przewidują etykietę binarną (0 = gorsze, 1 = lepsze).
    """

    sample = {
        "fixed acidity": 7.0,
        "volatile acidity": 0.27,
        "citric acid": 0.36,
        "residual sugar": 20.7,
        "chlorides": 0.045,
        "free sulfur dioxide": 45.0,
        "total sulfur dioxide": 170.0,
        "density": 1.0010,
        "pH": 3.00,
        "sulphates": 0.45,
        "alcohol": 8.8,
        "quality": 6,
    }

    df_sample = pd.DataFrame([sample])

    true_quality = df_sample["quality"].iloc[0]

    X_sample = df_sample.drop(columns=["quality"])

    X_sample_scaled = scaler.transform(X_sample)

    dt_pred = dt_model.predict(X_sample)[0]
    svm_pred = svm_model.predict(X_sample_scaled)[0]

    print("\n=== Przykładowa ręczna próbka ===")
    print("Wejście (cechy):")
    print(df_sample)
    print(f"Prawdziwa quality (skala 0-10): {true_quality}")
    print(f"Decision Tree przewiduje label : {dt_pred}")
    print(f"Best SVM (RBF) przewiduje      : {svm_pred}")


def plot_wine_correlation_heatmap(df):
    """
    Rysuje heatmapę korelacji dla całego zbioru winequality-white.

    Pokazuje zależności między wszystkimi 13 cechami fizykochemicznymi
    oraz kolumną 'label'. Pozwala ocenić, które cechy są powiązane
    z jakością wina (np. alcohol, residual sugar).
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
    plt.title("Wine Quality – Heatmap korelacji cech")
    plt.tight_layout()

    output_path = Path("output")
    output_path.mkdir(exist_ok=True)

    plt.savefig(output_path / "wine_correlation_heatmap.png", dpi=300)
    plt.show()

def main():
    df, X, y = load_winne_data(Path("data") / "winequality-white.csv")
    (
        X_train, X_test,
        y_train, y_test,
        X_train_scaled, X_test_scaled,
        scaler
    ) = split_and_scale(X, y)

    dt_model = train_decision_tree(X_train, y_train)

    #best_svm_model, grid = train_svm_with_gridsearch_all_kernels(X_train_scaled, y_train)
    svm_rbf_model = train_svm_rbf(X_train_scaled, y_train)

    evaluate_model("Decision Tree", dt_model, X_test, y_test)
    evaluate_model("Best SVM", svm_rbf_model, X_test_scaled, y_test)

    predict_custom_wine_sample(dt_model, svm_rbf_model, scaler)

    plot_feature_scatter(df)
    plot_wine_correlation_heatmap(df)

if __name__ == "__main__":
    main()
