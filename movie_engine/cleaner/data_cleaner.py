import pandas as pd

"""
Projekt: Silnik rekomendacji filmów/seriali – konwersja arkusza Excel do ratings.csv

Autorzy:
    Błażej Kanczkowski (s26836)
    Adam Rzepa (s27424)

Opis:
    Ten skrypt przekształca plik dataset.xlsx (format szeroki: user | title | rating | ...)
    na format ratings.csv wymagany przez silnik rekomendacji:
        user,title,rating

    W pliku Excel każdy wiersz ma strukturę:
        user, title1, rating1, title2, rating2, ...

Instrukcja przygotowania środowiska:
    1. Zainstaluj wymagane biblioteki:
         pip install pandas openpyxl

    2. Umieść plik dataset.xlsx w tym samym katalogu co skrypt.

    3. Uruchom:
         python convert_dataset.py

    4. Wynik zostanie zapisany do ratings.csv.

Summary:
    - wczytuje dataset.xlsx,
    - iteruje po wierszach,
    - paruje wartości (title, rating),
    - zapisuje ustrukturyzowany plik ratings.csv.

Returns:
    Tworzy plik ratings.csv w formacie zgodnym z całym projektem.
"""

df = pd.read_excel("dataset.xlsx", header=None)

rows = []
for i, row in df.iterrows():
    user = str(row.iloc[0]).strip()
    values = row.iloc[1:].dropna().tolist()
    for j in range(0, len(values), 2):
        try:
            title = str(values[j]).strip()
            rating = float(values[j+1])
        except Exception:
            continue
        rows.append((user, title, rating))

ratings = pd.DataFrame(rows, columns=["user", "title", "rating"])
print(ratings.head())

ratings.to_csv("ratings.csv", index=False)
