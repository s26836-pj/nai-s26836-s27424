from typing import List, Tuple

import unidecode

from models.cf_item import CFItemItem
from models.cbf_content import CBFContent

"""
Projekt: Silnik rekomendacji filmów/seriali – moduł Hybrid (CF + CBF)

Autorzy:
    Błażej Kanczkowski (s26836)
    Adam Rzepa (s27424)

Opis:
    Ten moduł implementuje hybrydowy system rekomendacji,
    który łączy dwa podejścia:
        - Collaborative Filtering (item–item)
        - Content-Based Filtering (TF-IDF na tytułach)

    Model Hybrid umożliwia:
        - generowanie rekomendacji przez połączenie sygnałów CF + CBF,
        - wyszukiwanie filmów podobnych do danego tytułu,
        - deduplikację tytułów (np. "Ex Machina" == "ex machina").

Instrukcja przygotowania środowiska:
    1. (Opcjonalnie) utwórz środowisko virtualenv:
         python -m venv venv
         source venv/bin/activate        # Linux/macOS
         venv\\Scripts\\activate         # Windows

    2. Zainstaluj biblioteki:
         pip install pandas scikit-learn unidecode

    3. Umieść plik ratings.csv w katalogu projektu:
         user,title,rating
"""

def norm_title(s: str) -> str:
    """
    Normalize a movie title for deduplication purposes.

    Summary:
        Converts title to lowercase, removes Polish diacritics
        and strips surrounding whitespace. Ensures that titles like
        'Ex Machina' and 'ex machina' are treated as the same key.

    Parameters:
        s (str):
            Original movie title.

    Returns:
        str:
            Normalized title string.
    """
    return unidecode.unidecode(str(s).lower().strip())


class Hybrid:
    """
       Hybrid recommendation engine combining CF and CBF results.

       Summary:
           Produces recommendations by merging:
               - Collaborative Filtering (CF item–item),
               - Content-Based Filtering (CBF TF-IDF textual similarity).

           The final score is a weighted linear combination of CF and CBF signals.
           Duplicate titles are removed using normalized title keys.
       """
    def __init__(self, ratings_csv: str, alpha: float = 0.6):
        """
               Initialize Hybrid model.

               Summary:
                   Loads ratings.csv, builds CF and CBF models, stores weight alpha.

               Parameters:
                   ratings_csv (str):
                       Path to ratings CSV (user,title,rating).
                   alpha (float):
                       Weight for CF contribution (0–1).
                       CBF weight is (1 - alpha).

               Returns:
                   None
               """
        self.cf = CFItemItem(ratings_csv)
        self.cbf = CBFContent.from_ratings_only(ratings_csv)
        self.alpha = alpha

    def _dedup_by_norm(
        self,
        items: List[Tuple[str, float]],
        k: int
    ) -> List[Tuple[str, float]]:
        seen_norm = set()
        out: List[Tuple[str, float]] = []
        """
                Deduplicate a list of (title, score) pairs using normalized titles.

                Summary:
                    Removes duplicated titles differing only in capital letters
                    or diacritics. Keeps the first (best scored) occurrence.

                Parameters:
                    items (List[Tuple[str, float]]):
                        List of (raw_title, score).
                    k (int):
                        Maximum number of results to keep.

                Returns:
                    List[Tuple[str, float]]:
                        List without duplicate normalized titles.
                """

        for title, score in items:
            nt = norm_title(title)
            if nt in seen_norm:
                continue
            seen_norm.add(nt)
            out.append((title, score))
            if len(out) >= k:
                break

        return out

    def similar_movies(self, title: str, k: int = 10):
        """
                Find movies similar to a given title using both CF and CBF signals.

                Summary:
                    Merges CF and CBF similarity lists and produces a weighted score:
                        score = alpha * cf_score + (1 - alpha) * cb_score
                    Then deduplicates the results.

                Parameters:
                    title (str):
                        Movie title to search similarities for.
                    k (int):
                        Number of similar titles to return.

                Returns:
                    List[Tuple[str, float]]:
                        List of (title, hybrid_score), sorted in descending order.
                """
        cf = dict(self.cf.similar_movies(title, k * 3))
        cb = dict(self.cbf.similar_movies(title, k * 3))
        keys = set(cf) | set(cb)

        merged = []
        for t in keys:
            score = self.alpha * cf.get(t, 0.0) + (1 - self.alpha) * cb.get(t, 0.0)
            merged.append((t, score))

        merged.sort(key=lambda x: x[1], reverse=True)

        return self._dedup_by_norm(merged, k)

    def recommend_for_user(
        self,
        user: str,
        k: int = 10,
        min_user_rating: float = 8.0
    ):
        """
         Produce hybrid recommendations for a single user.

        Summary:
            - CF provides a candidate list based on item–item similarity.
            - CBF finds movies similar to user’s favorite titles.
            - CF and CBF scores are merged linearly.
            - Titles already highly rated by the user are filtered out.
            - Final results are deduplicated by normalized title.

        Parameters:
            user (str):
                The user for whom recommendations are generated.
            k (int):
                Maximum number of recommendations to return.
            min_user_rating (float):
                Threshold for marking a title as "liked".

        Returns:
            List[Tuple[str, float]]:
                List of recommended (title, score) pairs.
        """
        cf_list = dict(self.cf.recommend_for_user(user, k * 3, min_user_rating))

        if user in self.cf.pivot.index:
            liked = [
                t for t, v in self.cf.pivot.loc[user].items()
                if v >= min_user_rating
            ]
        else:
            liked = []

        cb_scores = {}
        for base in liked[:5]:
            for t, sc in self.cbf.similar_movies(base, 20):
                cb_scores[t] = cb_scores.get(t, 0.0) + sc

        keys = set(cf_list) | set(cb_scores)
        merged = []
        for t in keys:
            score = (
                self.alpha * cf_list.get(t, 0.0)
                + (1 - self.alpha) * cb_scores.get(t, 0.0)
            )
            merged.append((t, score))

        already_high = {norm_title(t) for t in liked}
        merged = [
            (t, s) for (t, s) in merged
            if norm_title(t) not in already_high
        ]

        merged.sort(key=lambda x: x[1], reverse=True)

        return self._dedup_by_norm(merged, k)
