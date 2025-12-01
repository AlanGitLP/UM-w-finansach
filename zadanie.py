import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin

movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

sampled_users = np.random.choice(ratings["userId"].unique(), size=15000, replace=False)
ratings_sample = ratings[ratings["userId"].isin(sampled_users)]

sampled_movies = ratings_sample["movieId"].value_counts().head(15000).index
ratings_sample = ratings_sample[ratings_sample["movieId"].isin(sampled_movies)]

ratings_sample.reset_index(drop=True, inplace=True)

# Sci-Fi
sci_fi = movies[movies["genres"].str.contains("Sci-Fi", na=False)]["movieId"].nunique()
print("Liczba filmów Sci-Fi:", sci_fi)

# Komedie 2017
movies["year"] = movies["title"].str.extract(r"\((\d{4})\)").astype(float)
comedies_2017 = movies[
    (movies["genres"].str.contains("Comedy", na=False)) &
    (movies["year"] == 2017)
]["movieId"]

print("\nRozkład ocen komedii z 2017:")
print(ratings_sample[ratings_sample["movieId"].isin(comedies_2017)]["rating"].value_counts().sort_index())

# Filmy akcji
akcja = movies[movies["genres"].str.contains("Action", na=False)]
ratings_action = ratings_sample[ratings_sample["movieId"].isin(akcja["movieId"])]
print("\nŚrednia ocen filmów akcji:", ratings_action["rating"].mean())

top3 = ratings_action.groupby("movieId").size().sort_values(ascending=False).head(3)
top3 = top3.reset_index().merge(movies, on="movieId")[["movieId","title",0]]
print("\nTop 3 najczęściej oceniane filmy akcji:")
print(top3)

user_ids = ratings_sample["userId"].unique()
movie_ids = ratings_sample["movieId"].unique()

u_map = {u:i for i,u in enumerate(user_ids)}
m_map = {m:i for i,m in enumerate(movie_ids)}

R = np.zeros((len(user_ids), len(movie_ids)), dtype=np.float32)

for row in ratings_sample.itertuples():
    R[u_map[row.userId], m_map[row.movieId]] = row.rating

print("\nMacierz ocen:", R.shape)

similarity = cosine_similarity(R.T)
np.fill_diagonal(similarity, 0)

def recommend_knn(movie_title, top_n=10):
    movie_row = movies[movies["title"].str.contains(movie_title, case=False)]
    if movie_row.empty:
        return None
    mid = movie_row.iloc[0].movieId
    if mid not in m_map:
        return None
    idx = m_map[mid]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
    rec_ids = [movie_ids[i] for i,_ in scores]
    return movies[movies["movieId"].isin(rec_ids)][["movieId","title"]]

class BaselinePredictor(BaseEstimator, RegressorMixin):
    def fit(self, X, y=None):
        self.global_mean = R[R>0].mean()
        user_mean = R.sum(axis=1) / (R>0).sum(axis=1)
        item_mean = R.sum(axis=0) / (R>0).sum(axis=0)
        self.b_u = user_mean - self.global_mean
        self.b_i = item_mean - self.global_mean
        return self

    def predict(self, user_idx, movie_idx):
        return self.global_mean + self.b_u[user_idx] + self.b_i[movie_idx]

baseline = BaselinePredictor().fit(None)

svd = TruncatedSVD(n_components=50, random_state=42)
R_svd = svd.fit_transform(R)
R_reconstructed = np.dot(R_svd, svd.components_)

def recommend_svd(movie_title, top_n=10):
    movie_row = movies[movies["title"].str.contains(movie_title, case=False)]
    if movie_row.empty:
        return None
    mid = movie_row.iloc[0].movieId
    if mid not in m_map:
        return None
    idx = m_map[mid]
    movie_scores = R_reconstructed[:, idx]
    top_users = movie_scores.argsort()[::-1][:top_n]
    return top_users

def knn_score(k):
    neigh = NearestNeighbors(n_neighbors=k, metric="cosine")
    neigh.fit(R.T)
    distances, indices = neigh.kneighbors(R.T)
    # prosta metryka jakości – średnia odległość
    return distances.mean()

best_k = min(range(2,7), key=lambda k: knn_score(k))
print("\nNajlepsze k z przedziału 2–6:", best_k)

print("\nRekomendacje kNN dla Penguins of Madagascar:")
print(recommend_knn("Penguins of Madagascar"))

print("\nRekomendacje kNN dla Hobbit: The Desolation of Smaug:")
print(recommend_knn("Hobbit: The Desolation of Smaug"))