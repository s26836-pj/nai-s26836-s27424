import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import unidecode

"""
Projekt: Silnik rekomendacji filmów/seriali – moduł Content-Based Filtering (CBF)

Autorzy:
    Błażej Kanczkowski (s26836)
    Adam Rzepa (s27424)

Opis:
    Ten moduł implementuje prosty filtr typu content-based:
        - każdy film opisany jest tekstem (combined_text),
        - tekst wektorowany jest za pomocą TF-IDF (ngramy słowne),
        - podobieństwo między filmami liczone jest jako cosine similarity.

    Dzięki temu można:
        - szukać filmów podobnych treściowo do wskazanego tytułu,
        - zbudować hybrydowy system CF + CBF.

Instrukcja przygotowania środowiska:
    1. (Opcjonalnie) utwórz i aktywuj virtualenv:
         python -m venv venv
         source venv/bin/activate        # Linux/macOS
         venv\\Scripts\\activate         # Windows

    2. Zainstaluj wymagane biblioteki:
         pip install pandas scikit-learn unidecode

    3. Dane wejściowe:
         - DataFrame z kolumnami: title, combined_text
           lub plik ratings.csv z kolumnami: user,title,rating (dla from_ratings_only).
"""

def norm(s: str) -> str:
    """
       Normalize a text string for TF-IDF processing and comparison.

       Summary:
           Converts the string to lowercase, removes Polish diacritics
           and trims surrounding whitespace. This helps to:
               - reduce the impact of different capitalization,
               - unify characters with diacritics,
               - avoid leading/trailing spaces.

       Parameters:
           s (str):
               Raw input string (e.g. movie title or description).

       Returns:
           str:
               Normalized string ready for vectorization.
       """
    return unidecode.unidecode(str(s).lower().strip())

class CBFContent:
    """
       Content-Based Filtering model based on TF-IDF and cosine similarity.

       Summary:
           This class builds a content-based representation of movies by:
               - taking a DataFrame of movies (at least with 'title' and 'combined_text'),
               - computing a TF-IDF matrix over combined_text,
               - precomputing a cosine similarity matrix between all movies.

           It exposes a method to retrieve movies that are most similar in content
           to a given title.
       """
    def __init__(self, movies_df: pd.DataFrame):
        """
               Initialize the CBFContent model from a movies DataFrame.

               Summary:
                   - Removes duplicate titles.
                   - Builds a TF-IDF representation of the combined_text column.
                   - Precomputes cosine similarity between all movie vectors.

               Parameters:
                   movies_df (pd.DataFrame):
                       DataFrame containing at least:
                           - 'title' (str)          – movie title,
                           - 'combined_text' (str)  – textual description used for TF-IDF.

               Returns:
                   None
               """
        self.movies = movies_df.drop_duplicates("title").copy()
        self.movies["title_norm"] = self.movies["title"].map(norm)
        self.vec_word = TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=1)
        self.X = self.vec_word.fit_transform(self.movies["combined_text"].map(norm))
        self.sim = cosine_similarity(self.X)

    @classmethod
    def from_ratings_only(cls, ratings_csv: str):
        """
               Build a minimal CBFContent model using only movie titles from ratings.csv.

               Summary:
                   When no rich metadata (plot, genre, etc.) is available, this method
                   creates a simple content-based model where:
                       combined_text = title

                   It reads the ratings CSV, extracts unique titles and uses them
                   as both 'title' and 'combined_text'.

               Parameters:
                   ratings_csv (str):
                       Path to ratings.csv containing columns: user,title,rating.

               Returns:
                   CBFContent:
                       An instance of CBFContent initialized on the minimal dataset.
               """
        r = pd.read_csv(ratings_csv)
        movies = pd.DataFrame({"title": sorted(r["title"].unique())})
        movies["combined_text"] = movies["title"]
        return cls(movies)

    def similar_movies(self, title: str, k: int = 10):
        """
                Find movies most similar in content to a given title.

                Summary:
                    - Locates the movie index by exact title match.
                    - Takes the corresponding row from the cosine similarity matrix.
                    - Sorts all movies by similarity (descending).
                    - Skips the movie itself and returns top-k others.

                Parameters:
                    title (str):
                        Exact movie title present in self.movies['title'].
                    k (int):
                        Number of similar movies to return (default: 10).

                Returns:
                    List[Tuple[str, float]]:
                        A list of (title, similarity_score), sorted from most to
                        least similar. Returns an empty list if the title is unknown.
                """
        if title not in set(self.movies["title"]):
            return []
        idx = self.movies.index[self.movies["title"] == title][0]
        sims = list(enumerate(self.sim[idx]))
        sims = sorted(sims, key=lambda x: x[1], reverse=True)
        out = []
        for j, sc in sims:
            if j == idx:
                continue
            out.append((self.movies.iloc[j]["title"], float(sc)))
            if len(out) >= k:
                break
        return out
