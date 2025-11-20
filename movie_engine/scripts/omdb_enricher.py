"""
omdb_enricher.py — wzbogacanie danych filmów danymi z OMDb

Autor: Błażej Kanczkowski s26836 Adam Rzepa 27424

Opis:
    - czyta plik ../ratings.csv (user,title,rating),
    - wybiera ograniczony zbiór "interesujących" tytułów:
        * wszystkie filmy ocenione przez wybranego użytkownika,
        * + top N najczęściej ocenianych filmów globalnie,
    - dla tych tytułów pobiera dane z OMDb (z cache'em i limitem requestów),
    - zapisuje plik ../data/movies_meta_omdb.csv z kolumnami:
        title,year,genre,imdb_rating,plot,omdb_type,combined_text

Instrukcja użycia:
    1. Ustaw zmienną środowiskową OMDB_API_KEY z kluczem API do OMDb.
    Aby pobierać dane filmów, potrzebujesz bezpłatnego klucza API:

    Wejdź na stronę:
    -https://www.omdbapi.com/apikey.aspx
    -Wybierz Free plan.
    -Podaj e-mail i utwórz konto.
    -Otrzymasz swój OMDb API Key na maila (np. 123abc).

    Plik .env powinien znajdować sie /scripts
    2. Upewnij się, że ratings.csv jest w katalogu nadrzędnym.
    3. Uruchom:
         python omdb_enricher.py
    4. Następnie uruchom recs_clean.py, aby użyć metadanych w CBF.

Wymaga:
    pip install pandas requests unidecode
"""

import os
import json
import re
from pathlib import Path
from typing import List, Set

import pandas as pd
import requests
from unidecode import unidecode
from dotenv import load_dotenv

load_dotenv()

BASE = Path(__file__).resolve().parent
RATINGS_CSV = BASE.parent / "ratings.csv"
DATA_DIR = BASE.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = DATA_DIR / "movies_meta_omdb.csv"

CACHE_DIR = BASE.parent / "cache_omdb"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TARGET_USER = "Paweł Czapiewski"

MAX_REQUESTS_PER_RUN = 300

API_KEY = os.environ.get("OMDB_API_KEY")

def normalize_title_for_omdb(t: str) -> str:
    """
    Normalizacja tytułu pod kątem API OMDb:
    - usunięcie polskich znaków,
    - wycięcie fragmentów typu "sezon 2", "(2020)",
    - uproszczenie wielokrotnych spacji.
    """
    t = str(t)
    t = unidecode(t)
    t = re.sub(r"\bsezon\s*\d+\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*\(.*?\)\s*$", "", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t


def select_titles_to_enrich(ratings_csv: Path,
                            target_user: str,
                            top_n: int = 200) -> List[str]:
    """
    Wybierz ograniczony zbiór tytułów do wzbogacenia:
    - wszystkie filmy ocenione przez target_user,
    - plus top_n najczęściej ocenianych tytułów globalnie.
    """
    df = pd.read_csv(ratings_csv)
    df["title"] = df["title"].astype(str)

    user_titles: Set[str] = set(df[df["user"] == target_user]["title"])

    pop_titles = (
        df.groupby("title")["rating"]
          .count()
          .sort_values(ascending=False)
          .head(top_n)
          .index
    )
    pop_titles = set(pop_titles)

    titles = sorted(user_titles | pop_titles)
    return titles


def cache_path(title: str) -> Path:
    """
    Ścieżka do pliku cache dla danego tytułu (w formacie JSON).
    Używamy tytułu w wersji "bezpiecznej" (np. bez ukośników).
    """
    safe = title.replace("/", "_")
    return CACHE_DIR / f"{safe}.json"


def omdb_fetch(title: str, session: requests.Session,
               request_counter: dict):
    """
    Pobierz dane z OMDb dla danego tytułu:

    - najpierw sprawdza cache (cache_omdb),
    - jeśli nie ma, wysyła jedno zapytanie do OMDb (z limitem),
    - w przypadku przekroczenia limitu zwraca None,
    - wynik (pełen JSON) zapisuje do cache.
    """
    cp = cache_path(title)

    if cp.exists():
        with open(cp, encoding="utf-8") as f:
            return json.load(f)

    if request_counter["count"] >= MAX_REQUESTS_PER_RUN:
        print("Osiągnięto limit requestów w tym uruchomieniu. "
              "Pomijam dalsze zapytania do OMDb.")
        return None

    query_title = normalize_title_for_omdb(title)
    params = {
        "apikey": API_KEY,
        "t": query_title,
        "plot": "short",
        "r": "json",
    }

    try:
        r = session.get("http://www.omdbapi.com/", params=params, timeout=10)
        data = r.json()
    except Exception as exc:
        print(f"Błąd podczas zapytania OMDb dla '{title}': {exc}")
        return None

    request_counter["count"] += 1

    with open(cp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return data

def build_meta_csv():
    """
    Główna funkcja skryptu:
    - wybiera tytuły do wzbogacenia,
    - pobiera dane z OMDb (z limitem i cache),
    - zapisuje movies_meta_omdb.csv.
    """
    if not RATINGS_CSV.exists():
        raise RuntimeError(f"Nie znaleziono pliku {RATINGS_CSV}")

    titles = select_titles_to_enrich(RATINGS_CSV, TARGET_USER, top_n=200)
    print(f"Wybrano {len(titles)} tytułów do wzbogacenia.")

    rows = []
    counter = {"count": 0}

    with requests.Session() as session:
        for t in titles:
            print(f"OMDb: {t}")
            data = omdb_fetch(t, session, counter)
            if not data or data.get("Response") != "True":
                print("  Brak danych z OMDb, pomijam.")
                continue

            year = data.get("Year", "")
            genre = data.get("Genre", "")
            plot = data.get("Plot", "")
            imdb = data.get("imdbRating", "")
            omdb_type = data.get("Type", "")

            combined = f"{t} {genre} {year} {plot}".strip()

            rows.append({
                "title": t,
                "year": year,
                "genre": genre,
                "imdb_rating": imdb,
                "plot": plot,
                "omdb_type": omdb_type,
                "combined_text": combined,
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("Zapisano metadane do:", OUT_CSV)
    print("Liczba requestów OMDb w tym uruchomieniu:",
          counter["count"])


if __name__ == "__main__":
    if not API_KEY:
        raise RuntimeError(
            "Brak klucza OMDB_API_KEY w zmiennych środowiskowych. "
            "Ustaw np.: export OMDB_API_KEY=TWÓJ_KLUCZ"
        )
    build_meta_csv()


