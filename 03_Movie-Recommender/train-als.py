import pandas as pd #type: ignore
import numpy as np #type: ignore
from scipy import sparse #type: ignore

from sklearn.metrics import root_mean_squared_error #type: ignore
from sklearn.model_selection import train_test_split #type: ignore

from pathlib import Path
import os
import json


# --------------- Config -------------- 
# Hyperparameters
K = 11 # -> number of hidden features (in this case will be the different genres) 
lambda_reg = 0.1 # -> lambda regularization 


# Parameters
num_iterations = 20
# ----------------------------------------------



print("Program-Start:")
print(f"PARAMETERS: \nLambda = {lambda_reg} | K = {K} | Number of Training Iterations = {num_iterations}")

RATINGS = pd.read_csv(Path("data/ratings.csv"))
#MOVIES = pd.read_csv(Path("data/movies.csv")) #NOTE: Not currently being used.




# -------------- Data Preprocessing ----------------
# NOTE: V & V part 1: ensuring all indexes are categorically encoded properly. 
RATINGS = RATINGS.drop(columns = ['timestamp'])

RATINGS['user_code'] = RATINGS['userId'].astype('category').cat.codes
RATINGS['movie_code'] =  RATINGS['movieId'].astype('category').cat.codes

user_map = dict(enumerate(RATINGS['userId'].astype('category').cat.categories))
movie_map = dict(enumerate(RATINGS['movieId'].astype('category').cat.categories))

num_users = int(RATINGS['user_code'].nunique())
num_movies = int(RATINGS['movie_code'].nunique())

def generate_matrix(indices, ratings):
    row_indices = indices['user_code'].values
    col_indices = indices['movie_code'].values
    rating = ratings.values

    return sparse.coo_matrix(
        (rating, (row_indices, col_indices)), 
        shape=(num_users, num_movies)
    ).tocsr()

indicies_train, indicies_test, ratings_train, ratings_test = train_test_split(RATINGS[['user_code', 'movie_code']], RATINGS['rating'], test_size=0.2)

user_item_matrix = generate_matrix(indicies_train, ratings_train)
movie_user_matrix = user_item_matrix.tocsc()


train_row = indicies_train['user_code'].values
train_col = indicies_train['movie_code'].values
test_row = indicies_test['user_code'].values
test_col = indicies_test['movie_code'].values


U = np.random.normal(loc=0, scale=(1/np.sqrt(K)), size=(num_users, K)) #User matrix
V = np.random.normal(loc=0, scale=(1/np.sqrt(K)), size=(num_movies, K)) #Movie matrix
I = np.eye(K)



# ------------------------ TRAINING ---------------------------
print("----------------- Begin Training ----------------")
for training_iteration in range(num_iterations): 
    #fix V train U
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

    
    
    #fix U train V
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


    #RMSE calculation
    pred_ratings_train = (U[train_row] * V[train_col]).sum(axis=1)
    training_rmse = root_mean_squared_error(ratings_train, pred_ratings_train)

    pred_ratings_test = (U[test_row] * V[test_col]).sum(axis=1)
    test_rmse = root_mean_squared_error(ratings_test, pred_ratings_test)

    print(f"Iteration: {training_iteration + 1} | Train RMSE: {training_rmse} | Test RMSE: {test_rmse}\n")

print('----------------- End Training ----------------')


# ---------------------- Save to artifacts dir ---------------
os.makedirs("artifacts", exist_ok=True)

rating_frequency = (
    RATINGS['movieId']
    .value_counts()
    .rename_axis('movieId')
    .reset_index(name='count')
    .sort_values('count', ascending=False)
)
rating_frequency.to_csv("artifacts/rating_frequency.csv", index=False)
np.save("artifacts/User_Matrix", U)
np.save("artifacts/Movie_Matrix", V)
with open("artifacts/user_map.json", "w") as f:
    json.dump(user_map, f, indent=4)
with open("artifacts/movie_map.json", "w") as f:
    json.dump(movie_map, f, indent=4)
