# ML: Decision Tree & SVM – Wine Quality & Heart Disease Classification
## Autorzy
**Błażej Kanczkowski s26836**  
**Adam Rzepa s27424**

# Instalacja i uruchomienie projektu
Poniższe kroki pozwalają każdemu użytkownikowi uruchomić projekt lokalnie.

## Sklonowanie repozytorium

```bash
git clone https://github.com/<twoje-repo>/decision-tree-svm.git
cd decision-tree-svm
```

## Po sklonowaniu repozytorium przejdź do głównego katalogu projektu:

```bash
cd decision-tree-svm/decision_tree
```

## Utworzenie środowiska wirtualnego
Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

## Instalacja wymaganych bibliotek
```bash
pip install -r requirements.txt
```

## Pobranie danych
Wine Quality:
Plik ```winequality-white.csv``` znajduje się w katalogu:
```bash
data/winequality-white.csv
```

Heart Disease (Kaggle)
Dane są pobierane automatycznie przy uruchomieniu:
```bash
kagglehub.dataset_download("johnsmith88/heart-disease-dataset")
```

## Uruchomienie programu

Klasyfikacja Wine:
```bash
python decision_tree/wine_classification.py
```

Klasyfikacja Heart Disease:
```bash
python decision_tree/heart_classification.py
```

## Program automatycznie zapisze:

wine_scatter_alcohol_sugar.png

wine_correlation_heatmap.png

heart_correlation_heatmap.png

## W terminalu zobaczysz:
metryki modeli (Accuracy, Precision, Recall, F1),

najlepsze parametry z GridSearch,

predykcję pojedynczej próbki,

porównanie modeli.

##  Predykcja dla przykładowej, ręcznej próbki

Dodatkowo przygotowano funkcję ```predict_custom_wine_sample```, która demonstruje
działanie klasyfikatorów na ręcznie zdefiniowanej próbce wejściowej z cechami
w tym samym formacie jak w pliku ```winequality-white.csv```.