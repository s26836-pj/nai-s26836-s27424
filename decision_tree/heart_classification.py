"""
Projekt: Klasyfikacja chorób serca (Heart Disease) przy użyciu Decision Tree i SVM

Opis problemu:
Program wczytuje zbiór danych Heart Disease z platformy Kaggle
(automatycznie pobierany za pomocą biblioteki kagglehub),
a następnie wykonuje klasyfikację dwuklasową:
    0 – brak choroby serca
    1 – obecność choroby serca

W projekcie trenowane są dwa modele:
    - Drzewo decyzyjne (DecisionTreeClassifier)
    - Maszyna wektorów nośnych (SVM) z automatycznym doborem
      najlepszych parametrów i funkcji jądra (GridSearchCV)

Zawiera również:
    - Analizę korelacji cech (heatmap)
    - Predykcję dla przykładowej próbki z X_test
    - Obliczenie metryk klasyfikacji (Accuracy, Precision, Recall, F1)
    - Porównanie modeli

Autorzy:
    Błażej Kanczkowski (s26836)
    Adam Rzepa (s27424)

Instrukcja uruchomienia:
README.md
"""
import pandas as pd
from pathlib import Path
import kagglehub
import seaborn as sns
import matplotlib.pyplot as plt

from utils.ml_utils import (
    split_and_scale,
    train_decision_tree,
    train_svm_with_gridsearch_all_kernels,
    train_svm_rbf,
    evaluate_model,
    predict_example,
)

def load_heart_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Pobiera zbiór Heart Disease z Kaggle za pomocą kagglehub
    i zwraca (df, X, y).
    Zwracane etykiety są już w kolumnie 'target'.
    """

    print("Pobieranie danych z Kaggle...")
    path = kagglehub.dataset_download("johnsmith88/heart-disease-dataset")
    print("Pobrano dane. Path:", path)

    csv_path = Path(path) / "heart.csv"
    df = pd.read_csv(csv_path)

    X = df.drop(columns=["target"])
    y = df["target"]

    return df, X, y

def plot_heart_correlation_heatmap(df):
    """
    Rysuje heatmapę korelacji dla zbioru Heart Disease.

    Pokazuje zależności między wszystkimi cechami (np. age, trestbps,
    chol, thalach, oldpeak itd.) oraz kolumną 'target'.
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
    plt.title("Heart Disease – Heatmap korelacji cech")
    plt.tight_layout()

    output_path = Path("output")
    output_path.mkdir(exist_ok=True)

    plt.savefig(output_path / "heart_correlation_heatmap.png", dpi=300)
    plt.show()

def main():
    df, X, y = load_heart_dataset()

    (
        X_train, X_test,
        y_train, y_test,
        X_train_scaled, X_test_scaled,
        scaler
    ) = split_and_scale(X, y)

    dt_model = train_decision_tree(X_train, y_train)

    best_svm_model, grid = train_svm_with_gridsearch_all_kernels(X_train_scaled, y_train)

    evaluate_model("Decision Tree (heart)", dt_model, X_test, y_test)
    evaluate_model("Best SVM (z GridSearch – różne kernel)", best_svm_model, X_test_scaled, y_test)

    predict_example(5, X_test, y_test, scaler, dt_model, best_svm_model)

    plot_heart_correlation_heatmap(df)

if __name__ == "__main__":
    main()
