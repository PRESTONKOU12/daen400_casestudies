import pandas as pd
import numpy as np
from scipy import sparse

from pathlib import Path

# --------------- Config --------------
# Hyperparameters
K = 11 # -> number of hidden features (in this case will be the different genres)
lambda_reg = 0.1 # -> lambda regularization


# Parameters
num_iterations = 15
# ----------------------------------------------



RATINGS = pd.read_csv(Path("data/ratings.csv"))
MOVIES = pd.read_csv(Path("data/movies.csv"))


# -------------- Data Preprocessing ----------------
# NOTE: V & V part 1: ensuring all indexes are categorically encoded properly. 
RATINGS = RATINGS.drop(columns = ['timestamp'])

RATINGS['user_code'] = RATINGS['userId'].astype('category').cat.codes
RATINGS['movie_code'] =  RATINGS['movieId'].astype('category').cat.codes

user_map = dict(enumerate(RATINGS['userId'].astype('category').cat.categories))
movie_map = dict(enumerate(RATINGS['movieId'].astype('category').cat.categories))

row_indices = RATINGS['user_code'].values
col_indices = RATINGS['movie_code'].values
ratings = RATINGS['rating'].values

num_users = int(RATINGS['user_code'].nunique())
num_movies = int(RATINGS['movie_code'].nunique())

user_item_matrix = sparse.coo_matrix(
    (ratings, (row_indices, col_indices)), 
    shape=(num_users, num_movies)
).tocsr()
movie_user_matrix = user_item_matrix.tocsc()



U = np.random.normal(loc=0, scale=(1/np.sqrt(K)), size=(num_users, K)) #User matrix
V = np.random.normal(loc=0, scale=(1/np.sqrt(K)), size=(num_movies, K)) #Movie matrix
I = np.eye(K)



# ------------------------ TRAINING ---------------------------
for _ in range(1): #BUG: CHANGED NUM ITERATIONS TO MAGIC NUMBER 1 (Change back)
    #fix U train V
    for user in range(num_users):
        user_slice = user_item_matrix[user, :]
        rated_movie_indexes = user_slice.indices
        r_u = user_slice.data

        if len(rated_movie_indexes) == 0:
            continue #User has not rated any movies (skip)

        V_u = V[rated_movie_indexes, :]

        A = V_u.T @ V_u + lambda_reg * I

        b = V_u.T @ r_u 

        U[user] = np.linalg.solve(A, b)

    
    
    #fix V train U 
    for movie in range(num_movies):
        movie_slice = movie_user_matrix[:, movie]
        rated_user_index = movie_slice.indices
        r_m = movie_slice.data

        if len(rated_user_index) == 0:
            continue

        U_m = U[rated_user_index, :]

        A = U_m.T @ U_m + lambda_reg * I

        b = U_m.T @ r_m 

        V[movie] = np.linalg.solve(A, b)




#TODO: 
# - Move training loop into a function to pass in training split. 
# - Write RMSE tracker 
# - Integrate new users and get top 10 movie list.

