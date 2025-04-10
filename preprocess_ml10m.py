import pandas as pd
import numpy as np
import scipy.sparse as sp
import pickle
from sklearn.model_selection import train_test_split
import os

# Create the path to ratings.dat file
ratings_path = "Data/ml10m/ratings.dat"
ratings = pd.read_csv(ratings_path, sep="::", engine="python", 
                      names=["UserID", "MovieID", "Rating", "Timestamp"])

# Keep only records with rating >= 4
ratings = ratings[ratings["Rating"] >= 4]

# Remap user IDs and movie IDs to consecutive integers
user_ids = ratings["UserID"].unique()
movie_ids = ratings["MovieID"].unique()
user_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
movie_map = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
ratings["UserID"] = ratings["UserID"].map(user_map)
ratings["MovieID"] = ratings["MovieID"].map(movie_map)

# Split into training and test sets (80/20)
train_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=42)

# Build sparse matrices
num_users = len(user_ids)
num_movies = len(movie_ids)

trn_mat = sp.coo_matrix(
    (np.ones(len(train_ratings)), (train_ratings["UserID"], train_ratings["MovieID"])),
    shape=(num_users, num_movies)
)

tst_mat = sp.coo_matrix(
    (np.ones(len(test_ratings)), (test_ratings["UserID"], test_ratings["MovieID"])),
    shape=(num_users, num_movies)
)

# Create directory for data storage
os.makedirs("Data/ml10m/", exist_ok=True)

# Save sparse matrices
with open("Data/ml10m/trnMat.pkl", "wb") as f:
    pickle.dump(trn_mat, f)

with open("Data/ml10m/tstMat.pkl", "wb") as f:
    pickle.dump(tst_mat, f)

# Save ID mappings
with open("Data/ml10m/user_map.pkl", "wb") as f:
    pickle.dump(user_map, f)

with open("Data/ml10m/movie_map.pkl", "wb") as f:
    pickle.dump(movie_map, f)

print("Processing complete. Files and mappings saved to Data/ml10m/")
