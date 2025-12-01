from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import kagglehub

def split_and_scale(X, y, test_size: float = 0.2, random_state: int = 42):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler


def train_decision_tree(X_train, y_train, max_depth=None, random_state=42):
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def train_svm_with_gridsearch_all_kernels(X_train_scaled, y_train):
    """
    Grid Search po wielu funkcjach jądra (kernel):
    linear, rbf, sigmoid, poly.
    Zwraca najlepszy model spośród WSZYSTKICH kerneli.
    """

    param_grid = [
        {
            "kernel": ["linear"],
            "C": [1, 5, 10, 15]
        },
        {
            "kernel": ["rbf"],
            "C": [1, 5, 10, 15],
            "gamma": ["scale", 0.1, 0.01]
        },
        {
            "kernel": ["sigmoid"],
            "C": [1, 5, 10, 15],
            "gamma": ["scale", 0.1, 0.01]
        },
        {
            "kernel": ["poly"],
            "C": [1, 5, 10],
            "gamma": ["scale", 0.1],
            "degree": [2, 3, 4]
        }
    ]

    grid = GridSearchCV(
        estimator=SVC(probability=True, random_state=42),
        param_grid=param_grid,
        scoring="accuracy",
        cv=5,
        n_jobs=-1
    )

    grid.fit(X_train_scaled, y_train)

    print("WYNIK GLOBALNY - NAJLEPSZY DLA WSZYSTKICH KERNELI")
    print("Najlepszy kernel:", grid.best_params_["kernel"])
    print("Najlepsze parametry:", grid.best_params_)
    print("Najlepszy wynik CV:", grid.best_score_)

    return grid.best_estimator_, grid


def train_svm_rbf(X_train_scaled, y_train, c=5, gamma='scale'):
    model = SVC(
        kernel="rbf",
        C=c,
        gamma=gamma,
        probability=True,
        random_state=42,
        #class_weight = "balanced"
    )
    model.fit(X_train_scaled, y_train)
    return model


def evaluate_model(name: str, model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n{name}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))


def predict_example(
    idx: int,
    X_test,
    y_test,
    scaler: StandardScaler,
    dt_model,
    svm_model
):
    """
    Bierze jedną próbkę z X_test (po indeksie idx),
    pokazuje:
      - jej cechy,
      - prawdziwą etykietę,
      - predykcję drzewa,
      - predykcję SVM.
    """
    x_sample = X_test.iloc[[idx]]
    y_true = y_test.iloc[idx]

    x_sample_scaled = scaler.transform(x_sample)

    dt_pred = dt_model.predict(x_sample)[0]
    svm_pred = svm_model.predict(x_sample_scaled)[0]

    print("\nPojedyncza próbka")
    print("Index w X_test:", idx)
    print("Cechy próbki:")
    print(x_sample)
    print(f"Prawdziwa etykieta (label): {y_true}")
    print(f"Decision Tree przewiduje : {dt_pred}")
    print(f"SVM przewiduje : {svm_pred}")
