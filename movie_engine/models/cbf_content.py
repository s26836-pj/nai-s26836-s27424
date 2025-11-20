import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class ItemCF:
    def __init__(self, ratings_csv: str):
        self.ratings = pd.read_csv(ratings_csv)
        self.pivot = (self.ratings
                      .pivot_table(index="user", columns="title", values="rating")
                      .fillna(0))
        sim = cosine_similarity(self.pivot.T)
        self.movie_sim = pd.DataFrame(sim, index=self.pivot.columns, columns=self.pivot.columns)

    def similar_movies(self, title: str, k: int = 10):
        if title not in self.movie_sim.columns:
            return []
        s = self.movie_sim[title].sort_values(ascending=False)
        return [(t, float(v)) for t, v in s.iloc[1:k+1].items()]

    def recommend_for_user(self, user: str, k: int = 10, min_user_rating: float = 8.0):
        """Agreguje podobieństwa filmów wysoko ocenionych przez usera."""
        if user not in self.pivot.index:
            return []
        user_ratings = self.pivot.loc[user]
        liked = user_ratings[user_ratings >= min_user_rating].index.tolist()
        scores = {}
        for t in liked:
            for cand, sim in self.similar_movies(t, k=50):
                if cand in liked:
                    continue
                scores[cand] = scores.get(cand, 0.0) + sim
        out = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return out
