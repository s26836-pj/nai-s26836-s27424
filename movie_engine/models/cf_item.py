import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

"""
Projekt: Silnik rekomendacji filmów/seriali – moduł Collaborative Filtering (item–item)

Autorzy:
    Błażej Kanczkowski (s26836)
    Adam Rzepa (s27424)

Instrukcja przygotowania środowiska:
    1. (Opcjonalnie) Utwórz środowisko virtualenv:
         python -m venv venv
         source venv/bin/activate     # Linux/macOS
         venv\\Scripts\\activate      # Windows

    2. Zainstaluj wymagane biblioteki:
         pip install pandas scikit-learn

    3. Umieść plik ratings.csv obok skryptu. Struktura:
         user,title,rating
"""

class CFItemItem:
    """
     Collaborative Filtering (item–item) using cosine similarity.

     Summary:
         This class computes item–item similarity based on a user×title
         rating matrix (pivot table). It allows:
             - retrieving movies similar to a given title,
             - generating recommendations for a specific user.

         The method is based on the assumption that two items are similar
         if they were rated similarly by many users.
     """
    def __init__(self, ratings_csv: str):
        """
              Initialize the CFItemItem model from a ratings CSV file.

              Parameters:
                  ratings_csv (str):
                      Path to a CSV file containing user ratings.
                      Required columns: user, title, rating.

              Returns:
                  None
              """
        self.ratings = pd.read_csv(ratings_csv)
        self.pivot = (self.ratings
                      .pivot_table(index="user", columns="title", values="rating")
                      .fillna(0))
        sim = cosine_similarity(self.pivot.T)
        self.movie_sim = pd.DataFrame(sim, index=self.pivot.columns, columns=self.pivot.columns)

    def similar_movies(self, title: str, k: int = 10):
        """
               Find k movies most similar to the given title.

               Summary:
                   Uses cosine similarity matrix to return the top-k similar movies.
                   The movie itself is always skipped in the output.

               Parameters:
                   title (str):
                       Title of the movie for which similar items should be found.
                   k (int):
                       Number of similar movies to return (default: 10).

               Returns:
                   List[Tuple[str, float]]:
                       A list of (movie_title, similarity_score), sorted by score
                       in descending order. Returns an empty list if the title is
                       not known in the dataset.
               """
        if title not in self.movie_sim.columns:
            return []
        s = self.movie_sim[title].sort_values(ascending=False)
        return [(t, float(v)) for t, v in s.iloc[1:k+1].items()]  # pomiń sam film

    def recommend_for_user(self, user: str, k: int = 10, min_user_rating: float = 8.0, bad_rating: float = 4.0):
        """
        Generate item–item collaborative filtering recommendations for a user.

        Summary:
            Produces a ranked list of recommended movies based on:
                - high-rated movies (>= min_user_rating),
                - penalizing movies similar to disliked ones (<= bad_rating),
                - skipping already rated movies.
            Similarity scores are weighted by how much a user liked a movie.

        Parameters:
            user (str):
                User identifier (must exist in the pivot table).
            k (int):
                Maximum number of recommendations to return.
            min_user_rating (float):
                Threshold above which a movie is considered "liked".
            bad_rating (float):
                Threshold below which a movie is considered "disliked".

        Returns:
            List[Tuple[str, float]]:
                List of recommended movies as (title, weighted_score),
                sorted from most to least relevant.
                Returns empty list if user is unknown.
        """

        if user not in self.pivot.index:
            return []

        user_ratings = self.pivot.loc[user]

        liked = user_ratings[user_ratings >= min_user_rating].index.tolist()
        disliked = user_ratings[user_ratings <= bad_rating].index.tolist()
        already_rated = user_ratings[user_ratings > 0].index.tolist()

        scores = {}

        for t in liked:
            rating = user_ratings[t]
            weight = (rating - min_user_rating + 1)  # np. 8→1, 10→3
            for cand, sim in self.similar_movies(t, k=50):
                if cand in already_rated:
                    continue
                if cand in disliked:
                    continue
                scores[cand] = scores.get(cand, 0.0) + sim * weight

        for t in disliked:
            for cand, sim in self.similar_movies(t, k=50):
                if cand in scores:
                    scores[cand] -= sim * 2

        out = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return out

