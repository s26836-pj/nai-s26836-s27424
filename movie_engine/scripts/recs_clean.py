"""
recs_clean.py — Silnik rekomendacji filmów/seriali

Autorzy:
    Błażej Kanczkowski (s26836)
    Adam Rzepa (s27424)

Opis problemu:
    Dane wejściowe:
        - plik ratings.csv z kolumnami: user,title,rating
        - (opcjonalnie) plik data/movies_meta_omdb.csv z metadanymi z OMDb
    Celem jest zbudowanie silnika rekomendacji, który:
        - rekomenduje filmy dla wybranego użytkownika (CF + CBF + klastrowanie),
        - generuje antyrekomendacje (czego raczej nie oglądać),
        - dołącza dodatkowe informacje o filmie (rok, gatunek, opis)
          z pliku movies_meta_omdb.csv (dane z zewnętrznego API OMDb).

Metody:
    - CF (item–item):
        podobieństwo cosinusowe między filmami na podstawie ocen (user×title).
    - CBF (content-based):
        TF-IDF po tekście opisującym film (combined_text).
    - Clustering użytkowników:
        TruncatedSVD + KMeans na macierzy user×film z progami liczby
        i jakości ocen w klastrze.
    - Hybrid:
        połączenie sygnałów CF + CBF + klaster (Hybrid3).

Instrukcja przygotowania środowiska:
    1. (Opcjonalnie) utwórz i aktywuj virtualenv:
         python -m venv venv
         source venv/bin/activate        # Linux/macOS
         venv\\Scripts\\activate         # Windows

    2. Zainstaluj wymagane biblioteki:
         pip install pandas numpy scikit-learn unidecode requests

    3. Przygotuj dane:
         - ratings.csv w katalogu nadrzędnym (obok tego pliku),
         - (opcjonalnie) uruchom omdb_enricher.py, aby utworzyć:
             data/movies_meta_omdb.csv

    4. Uruchom silnik rekomendacji:
         python recs_clean.py

    5. Skrypt wypisze:
         - rekomendacje dla wybranego użytkownika,
         - antyrekomendacje,
         - jeżeli dostępne: dodatkowe informacje o filmach z OMDb.

Summary:
    Hybrid recommendation engine that combines CF, CBF and user clustering.
    This file can be treated as the main entry point of the project.

Returns:
    Po uruchomieniu jako skrypt: wypisuje rekomendacje i antyrekomendacje
    w konsoli; nie zwraca wartości z poziomu modułu.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from unidecode import unidecode


def norm(s: str) -> str:
    """
    Normalize a text string for internal comparison.

    Summary:
        Converts the string to lowercase, removes Polish diacritics
        and trims surrounding whitespace. Used for:
            - deduplication of titles,
            - text preprocessing before TF-IDF.

    Parameters:
        s (str):
            Raw input string (e.g. movie title).

    Returns:
        str:
            Normalized string.
    """
    return unidecode(str(s).lower().strip())


def looks_like_same_series(a: str, b: str) -> bool:
    """
    Check if two titles look like they belong to the same series.

    Summary:
        A simple heuristic used for sequels detection, e.g.:
            "1670" vs "1670 sezon 2"
        It removes parts after ":" and occurrences of the word "sezon"
        with a number, then compares the base titles.

    Parameters:
        a (str):
            First title.
        b (str):
            Second title.

    Returns:
        bool:
            True if both titles share the same base string (series),
            False otherwise.
    """
    a2, b2 = a.lower(), b.lower()
    base_a = a2.split(":")[0].split("sezon")[0].strip()
    base_b = b2.split(":")[0].split("sezon")[0].strip()
    return bool(base_a) and base_a == base_b


def safe_text(x) -> str:
    """
    Convert a value from a DataFrame into a safe string.

    Summary:
        Handles NaN/None and non-string values in a robust way so that
        operations like len() do not fail on floats or missing values.

    Parameters:
        x (Any):
            Value read from a DataFrame cell (may be NaN, float, etc.).

    Returns:
        str:
            Empty string for NaN/None, otherwise string representation of x.
    """
    if isinstance(x, str):
        return x
    if pd.isna(x):
        return ""
    return str(x)


class CFItemItem:
    """
    Collaborative Filtering (item–item) based on cosine similarity.

    Summary:
        Builds a user×title rating matrix from ratings.csv and computes
        an item–item cosine similarity matrix. Provides:
            - similar_movies(): list of similar titles,
            - recommend_for_user(): CF-only recommendations.
    """

    def __init__(self, ratings_csv: str):
        """
        Initialize CFItemItem model from ratings.csv.

        Summary:
            - loads ratings,
            - builds user×title pivot table,
            - computes cosine similarity between titles.

        Parameters:
            ratings_csv (str):
                Path to ratings.csv containing columns: user,title,rating.

        Returns:
            None
        """
        r = pd.read_csv(ratings_csv)
        r["title"] = r["title"].astype(str)

        self.pivot = (
            r.pivot_table(index="user", columns="title", values="rating")
             .fillna(0)
        )

        sim = cosine_similarity(self.pivot.T)
        self.movie_sim = pd.DataFrame(
            sim,
            index=self.pivot.columns,
            columns=self.pivot.columns
        )

    def similar_movies(self, title: str, k: int = 10):
        """
        Return k movies most similar to a given title.

        Summary:
            Uses the precomputed cosine similarity matrix to return
            nearest neighbours for the given title. The movie itself
            is not included in the result list.

        Parameters:
            title (str):
                Title for which similar movies are requested.
            k (int):
                Number of similar titles to return (default: 10).

        Returns:
            List[Tuple[str, float]]:
                List of (title, similarity_score) sorted in descending
                order of similarity. For unknown titles – empty list.
        """
        if title not in self.movie_sim:
            return []
        s = self.movie_sim[title].sort_values(ascending=False)
        return [(t, float(v)) for t, v in s.iloc[1:k + 1].items()]

    def recommend_for_user(
        self,
        user: str,
        k: int = 10,
        min_user_rating: float = 8.0,
        bad_rating: float = 4.0
    ):
        """
        Generate CF-based recommendations for a user.

        Summary:
            - considers movies with rating >= min_user_rating as liked,
            - considers movies with rating <= bad_rating as disliked,
            - skips all already rated movies in the output,
            - penalizes candidates similar to disliked titles.

        Parameters:
            user (str):
                User identifier present in the pivot index.
            k (int):
                Maximum number of recommended titles to return.
            min_user_rating (float):
                Threshold for treating a rating as "liked".
            bad_rating (float):
                Threshold for treating a rating as "disliked".

        Returns:
            List[Tuple[str, float]]:
                List of (title, score) sorted from highest to lowest score.
                Returns an empty list if user is unknown.
        """
        if user not in self.pivot.index:
            return []

        u = self.pivot.loc[user]
        liked = u[u >= min_user_rating].index.tolist()
        disliked = u[u <= bad_rating].index.tolist()
        seen = set(u[u > 0].index)

        scores: dict[str, float] = {}

        # positive contribution from liked movies
        for t in liked:
            weight = (u[t] - min_user_rating + 1.0)  # e.g. 8 → 1, 10 → 3
            for cand, sim in self.similar_movies(t, 30):
                if cand in seen or cand in disliked:
                    continue
                scores[cand] = scores.get(cand, 0.0) + sim * weight

        # negative contribution from disliked movies
        for t in disliked:
            for cand, sim in self.similar_movies(t, 30):
                if cand in scores:
                    scores[cand] -= sim * 2.0

        out = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        out = [(t, s) for t, s in out if t not in seen]
        return out[:k]


class CBFContent:
    """
    Content-Based Filtering model using TF-IDF and cosine similarity.

    Summary:
        Represents each movie by its combined_text (e.g. title + genre + plot),
        vectorizes it with TF-IDF and precomputes cosine similarity between
        all movies. Can be built either from ratings (titles only) or from
        external metadata (OMDb) with richer combined_text.
    """

    def __init__(self, movies_df: pd.DataFrame):
        """
        Initialize CBFContent from a movies DataFrame.

        Summary:
            - removes duplicate titles,
            - fills missing combined_text with title,
            - fits TF-IDF vectorizer,
            - computes cosine similarity matrix.

        Parameters:
            movies_df (pd.DataFrame):
                DataFrame with at least two columns:
                    - 'title' (str),
                    - 'combined_text' (str).

        Returns:
            None
        """
        self.movies = movies_df.drop_duplicates("title").copy()
        self.movies["combined_text"] = self.movies["combined_text"].fillna(
            self.movies["title"]
        )

        self.vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1
        )
        self.X = self.vec.fit_transform(self.movies["combined_text"].map(norm))
        self.sim = cosine_similarity(self.X)

    @classmethod
    def from_ratings(cls, ratings_csv: str):
        """
        Build a minimal CBFContent model using only ratings.csv.

        Summary:
            Constructs a DataFrame with unique titles from ratings.csv
            and sets combined_text = title. Used as a fallback when
            no external metadata is available.

        Parameters:
            ratings_csv (str):
                Path to ratings.csv containing columns: user,title,rating.

        Returns:
            CBFContent:
                Instance of CBFContent built from title-only data.
        """
        r = pd.read_csv(ratings_csv)
        movies = pd.DataFrame({"title": sorted(r["title"].astype(str).unique())})
        movies["combined_text"] = movies["title"]
        return cls(movies)

    @classmethod
    def from_metadata(cls, meta_csv: str):
        """
        Build CBFContent model from external metadata (e.g. OMDb).

        Summary:
            Creates a DataFrame from a metadata CSV file containing at least:
                - title
                - combined_text
            and uses it to build a content-based model with TF-IDF.

        Parameters:
            meta_csv (str):
                Path to a CSV file with metadata, including 'title'
                and 'combined_text' columns.

        Returns:
            CBFContent:
                Instance of CBFContent built from external metadata.
        """
        movies = pd.read_csv(meta_csv)
        movies = movies.dropna(subset=["title"]).copy()
        movies["combined_text"] = movies["combined_text"].fillna(movies["title"])
        return cls(movies)

    def similar_movies(self, title: str, k: int = 10):
        """
        Return movies that are most similar in content to the given title.

        Summary:
            - finds the index of the given title,
            - sorts all movies by cosine similarity to it,
            - skips the movie itself,
            - returns top-k matches.

        Parameters:
            title (str):
                Exact title present in self.movies['title'].
            k (int):
                Number of similar movies to return.

        Returns:
            List[Tuple[str, float]]:
                List of (title, similarity_score). Empty list if unknown title.
        """
        titles = set(self.movies["title"])
        if title not in titles:
            return []
        idx = self.movies.index[self.movies["title"] == title][0]
        sims = list(enumerate(self.sim[idx]))
        sims.sort(key=lambda x: x[1], reverse=True)

        out = []
        for j, sc in sims:
            if j == idx:
                continue
            out.append((self.movies.iloc[j]["title"], float(sc)))
            if len(out) >= k:
                break
        return out


class UserClusters:
    """
    User clustering and cluster-based recommendations.

    Summary:
        Applies dimensionality reduction (TruncatedSVD) and KMeans clustering
        on the user×title rating matrix. For each cluster, stores:
            - mean rating per title,
            - count of ratings per title,
            - share of high ratings per title.
        Provides:
            - recommend_for_user(): cluster-based recommendations,
            - anti_for_user(): cluster-based anti-recommendations.
    """

    def __init__(
        self,
        pivot: pd.DataFrame,
        n_clusters: int = 5,
        svd_dim: int = 20,
        high_rating: float = 8.0
    ):
        """
        Initialize UserClusters model from a user×title pivot.

        Summary:
            - reduces dimensionality of the pivot using TruncatedSVD,
            - normalizes embeddings and runs KMeans,
            - computes per-cluster statistics:
                mean rating, rating count, share of high ratings.

        Parameters:
            pivot (pd.DataFrame):
                User×title rating matrix (0 means "no rating").
            n_clusters (int):
                Number of KMeans clusters.
            svd_dim (int):
                Target dimension for SVD-reduced space.
            high_rating (float):
                Threshold for counting a rating as "high" (e.g. >= 8.0).

        Returns:
            None
        """
        self.pivot = pivot
        self.high_rating = high_rating

        X = pivot.values
        svd_dim = min(svd_dim, max(2, min(X.shape) - 1))
        Xs = TruncatedSVD(n_components=svd_dim, random_state=42).fit_transform(X)
        Xn = normalize(Xs)
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        self.labels = self.kmeans.fit_predict(Xn)
        self.cluster_by_user = dict(zip(pivot.index, self.labels))

        df = pivot.replace(0, np.nan).copy()
        df["__cluster"] = self.labels

        self.cluster_item_mean = df.groupby("__cluster").mean(numeric_only=True)
        self.cluster_item_count = df.groupby("__cluster").count()

        is_high = df.drop(columns="__cluster").ge(self.high_rating)
        is_high["__cluster"] = self.labels
        high_count = is_high.groupby("__cluster").sum(numeric_only=True)
        self.cluster_item_high_share = (
            high_count / self.cluster_item_count
        ).fillna(0.0)

    def _cluster_candidates(
        self,
        user: str,
        k: int,
        min_count: int,
        min_high_share: float,
        reverse: bool
    ):
        """
        Internal helper to get cluster-based candidate titles for a user.

        Summary:
            - finds the user's cluster,
            - filters titles with at least min_count ratings,
            - optionally filters by share of high ratings (min_high_share),
            - sorts by mean rating (ascending or descending).

        Parameters:
            user (str):
                User identifier.
            k (int):
                Number of titles to return.
            min_count (int):
                Minimum number of ratings in the cluster.
            min_high_share (float):
                Minimum share of high ratings (if reverse is False).
            reverse (bool):
                If True – sort ascending (for anti-recommendations),
                if False – sort descending (for recommendations).

        Returns:
            List[Tuple[str, float]]:
                List of (title, mean_rating_in_cluster).
        """
        if user not in self.pivot.index:
            return []

        cl = self.cluster_by_user[user]
        seen = set(self.pivot.loc[user][self.pivot.loc[user] > 0].index)

        mean = self.cluster_item_mean.loc[cl]
        cnt = self.cluster_item_count.loc[cl]
        share = self.cluster_item_high_share.loc[cl]

        ok = (cnt >= min_count)
        if not reverse:
            ok = ok & (share >= min_high_share)

        mean = mean[ok].drop(labels=list(seen), errors="ignore")
        mean = mean.sort_values(ascending=reverse)

        return [(t, float(v)) for t, v in mean.head(k).items()]

    def recommend_for_user(
        self,
        user: str,
        k: int = 10,
        min_count: int = 2,
        min_high_share: float = 0.4
    ):
        """
        Generate cluster-based recommendations for a user.

        Summary:
            - selects titles not yet watched by the user,
            - requires at least min_count ratings in the cluster,
            - requires share of high ratings >= min_high_share.

        Parameters:
            user (str):
                User identifier.
            k (int):
                Number of recommendations to return.
            min_count (int):
                Minimum number of ratings per title in the cluster.
            min_high_share (float):
                Minimum share of high ratings per title.

        Returns:
            List[Tuple[str, float]]:
                List of (title, mean_cluster_rating).
        """
        return self._cluster_candidates(
            user,
            k,
            min_count,
            min_high_share,
            reverse=False
        )

    def anti_for_user(self, user: str, k: int = 10, min_count: int = 1):
        """
        Generate cluster-based anti-recommendations for a user.

        Summary:
            - selects titles not yet watched by the user,
            - requires at least min_count ratings in the cluster,
            - picks titles with the lowest mean rating in the cluster,
            - skips sequels of series that the user likes (high_rating threshold).

        Parameters:
            user (str):
                User identifier.
            k (int):
                Number of anti-recommendations to return.
            min_count (int):
                Minimum number of ratings in the cluster.

        Returns:
            List[Tuple[str, float]]:
                List of (title, mean_cluster_rating) with low scores.
        """
        raw = self._cluster_candidates(
            user,
            k * 3,
            min_count,
            min_high_share=0.0,
            reverse=True
        )

        liked = [t for t, v in self.pivot.loc[user].items()
                 if v >= self.high_rating]
        out = []
        for t, s in raw:
            if any(looks_like_same_series(t, L) for L in liked):
                continue
            out.append((t, s))
            if len(out) >= k:
                break
        return out


class Hybrid3:
    """
    Hybrid recommendation engine combining CF, CBF and user clusters.

    Summary:
        For a given user, this class:
            - gathers CF candidates (item–item),
            - gathers CBF candidates (similar to liked movies),
            - gathers cluster-based candidates (UserClusters),
            - merges all scores using weights (alpha_cf, beta_cbf, gamma_cluster),
            - filters out titles already seen by the user,
            - deduplicates normalized titles.
        Also delegates anti-recommendations to UserClusters.
    """

    def __init__(
        self,
        ratings_csv: str,
        alpha_cf: float = 0.5,
        beta_cbf: float = 0.3,
        gamma_cluster: float = 0.2
    ):
        """
        Initialize Hybrid3 model from ratings.csv.

        Summary:
            - builds CFItemItem from ratings.csv,
            - tries to load CBFContent from metadata (movies_meta_omdb.csv);
              falls back to CBFContent.from_ratings if metadata is missing,
            - builds UserClusters based on CF pivot,
            - stores weights for CF/CBF/cluster components.

        Parameters:
            ratings_csv (str):
                Path to ratings.csv.
            alpha_cf (float):
                Weight of CF component in hybrid score.
            beta_cbf (float):
                Weight of CBF component in hybrid score.
            gamma_cluster (float):
                Weight of cluster component in hybrid score.

        Returns:
            None
        """
        self.cf = CFItemItem(ratings_csv)

        ratings_path = Path(ratings_csv)
        meta_csv = ratings_path.parent / "data" / "movies_meta_omdb.csv"
        if meta_csv.exists():
            self.cbf = CBFContent.from_metadata(str(meta_csv))
        else:
            self.cbf = CBFContent.from_ratings(ratings_csv)

        self.cl = UserClusters(self.cf.pivot)
        self.alpha = alpha_cf
        self.beta = beta_cbf
        self.gamma = gamma_cluster

    def recommend_for_user(self, user: str, k: int = 10):
        """
        Generate hybrid recommendations for a user.

        Summary:
            1. Collects candidates from:
                - CF recommendations,
                - CBF around user's liked titles,
                - cluster-based recommendations.
            2. Merges scores linearly:
                hybrid_score = base_bias + alpha*cf + beta*cbf + gamma*cluster
            3. Filters out titles already seen by the user.
            4. Deduplicates normalized titles and returns best k.

        Parameters:
            user (str):
                User identifier.
            k (int):
                Maximum number of recommendations to return.

        Returns:
            List[Tuple[str, float]]:
                List of (title, hybrid_score) sorted descending.
        """
        seen = set()
        if user in self.cf.pivot.index:
            seen = set(
                self.cf.pivot.loc[user][self.cf.pivot.loc[user] > 0].index
            )

        cf_scores = dict(self.cf.recommend_for_user(user, k * 3))
        liked = [t for t, v in self.cf.pivot.loc[user].items()
                 if v >= 8.0] if user in self.cf.pivot.index else []

        # CBF candidates around liked titles
        cb_scores: dict[str, float] = {}
        for base in liked[:5]:
            for t, sc in self.cbf.similar_movies(base, 20):
                if t in seen:
                    continue
                cb_scores[t] = cb_scores.get(t, 0.0) + sc

        cl_scores = dict(self.cl.recommend_for_user(user, k * 3))

        all_keys = (
            set(cf_scores) | set(cb_scores) | set(cl_scores)
        ) - seen
        out = []

        base_bias = float(self.cl.cluster_item_mean.mean().mean()) / 10.0

        for t in all_keys:
            score = base_bias + (
                self.alpha * cf_scores.get(t, 0.0)
                + self.beta * cb_scores.get(t, 0.0)
                + self.gamma * cl_scores.get(t, 0.0)
            )
            if score <= 0.0001:
                continue
            out.append((t, score))

        # Deduplicate by normalized title, keep best score
        best_by_norm: dict[str, tuple[str, float]] = {}
        for title, score in out:
            key = norm(title)
            if key not in best_by_norm or score > best_by_norm[key][1]:
                best_by_norm[key] = (title, score)

        deduped = list(best_by_norm.values())
        deduped.sort(key=lambda x: x[1], reverse=True)
        return deduped[:k]

    def anti_for_user(self, user: str, k: int = 10):
        """
        Generate hybrid anti-recommendations for a user.

        Summary:
            Delegates the logic to the cluster-based anti-recommendations
            from UserClusters (lowest mean ratings in user's cluster).

        Parameters:
            user (str):
                User identifier.
            k (int):
                Number of anti-recommendations to return.

        Returns:
            List[Tuple[str, float]]:
                List of (title, mean_cluster_rating).
        """
        return self.cl.anti_for_user(user, k)


class MovieInfo:
    """
    Helper class for loading movie metadata.

    Summary:
        Reads data/movies_meta_omdb.csv (if present) and stores rows
        in a dictionary keyed by title. Used for enriching printed
        recommendations and anti-recommendations with year, genre and plot.
    """

    def __init__(self, meta_csv: str):
        """
        Initialize MovieInfo with a metadata CSV file.

        Summary:
            Loads metadata (if file exists) into an internal dictionary
            accessible by movie title.

        Parameters:
            meta_csv (str):
                Path to movies_meta_omdb.csv.

        Returns:
            None
        """
        self.by_title: dict[str, pd.Series] = {}
        p = Path(meta_csv)
        if p.exists():
            df = pd.read_csv(p)
            for _, row in df.iterrows():
                self.by_title[str(row["title"])] = row

    def get(self, title: str):
        """
        Get metadata row for a given title.

        Summary:
            Looks up the internal dictionary and returns the corresponding
            pandas Series with metadata fields (year, genre, plot, etc.).

        Parameters:
            title (str):
                Movie title.

        Returns:
            pandas.Series | None:
                Metadata row if present, otherwise None.
        """
        return self.by_title.get(title)


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    ratings_path = base.parent / "ratings.csv"
    meta_path = base.parent / "data" / "movies_meta_omdb.csv"

    model = Hybrid3(ratings_csv=str(ratings_path))
    info = MovieInfo(str(meta_path))

    user = "Błażej Kanczkowski"

    print("\nRekomendacje dla użytkownika:", user)
    recs = model.recommend_for_user(user, 5)
    for title, score in recs:
        meta = info.get(title)
        if meta is not None:
            year = safe_text(meta.get("year", ""))
            genre = safe_text(meta.get("genre", ""))
            plot = safe_text(meta.get("plot", ""))
            if len(plot) > 160:
                plot = plot[:160] + "..."
            print(f"  + {title} ({year}) [{genre}]  score={score:.3f}")
            if plot:
                print(f"      {plot}")
        else:
            print(f"  + {title}  score={score:.3f} (brak danych OMDb)")

    print("\nAntyrekomendacje dla użytkownika:", user)
    anti = model.anti_for_user(user, 5)

    if not anti:
        print(
            "  (brak wystarczających danych w klastrze, "
            "żeby zbudować antyrekomendacje)"
        )
    else:
        for title, score in anti:
            meta = info.get(title)
            if meta is not None:
                year = safe_text(meta.get("year", ""))
                genre = safe_text(meta.get("genre", ""))
                print(f"  - {title} ({year}) [{genre}]  score={score:.3f}")
            else:
                print(
                    f"  - {title}  score={score:.3f} (brak danych OMDb)"
                )

