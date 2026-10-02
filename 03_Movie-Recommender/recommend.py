# recommend.py
import json
import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ===================== artifact loading =====================

def load_artifacts():
    U = np.load("artifacts/User_Matrix.npy")
    V = np.load("artifacts/Movie_Matrix.npy")

    with open("artifacts/movie_map.json", "r") as f:
        movie_map = json.load(f)

    with open("artifacts/user_map.json", "r") as f:
        user_map = json.load(f)

    rating_frequency = pd.read_csv("artifacts/rating_frequency.csv")
    movies = pd.read_csv("data/movies.csv")

    return U, V, movie_map, user_map, rating_frequency, movies


U, V, movie_map, user_map, rating_frequency, movies_df = load_artifacts()

# Build reverse lookup: movieId -> movie_code (int)
movie_id_to_code = {int(v): int(k) for k, v in movie_map.items()}

# Merge titles onto rating_frequency
rating_frequency = rating_frequency.merge(
    movies_df[['movieId', 'title', 'genres']], on='movieId', how='left'
)

# Pre-compute the set of "recommendable" movies:
# only movies with a meaningful number of ratings
MIN_RATINGS_TO_RECOMMEND = 20
recommendable_movie_ids = set(
    rating_frequency.loc[
        rating_frequency['count'] >= MIN_RATINGS_TO_RECOMMEND, 'movieId'
    ].tolist()
)
recommendable_codes = [
    int(k) for k, v in movie_map.items()
    if int(v) in recommendable_movie_ids
]
recommendable_codes_arr = np.array(recommendable_codes)
# Pre-slice V for fast scoring
V_recommendable = V[recommendable_codes_arr, :]
recommendable_mids = np.array([int(movie_map[str(c)]) for c in recommendable_codes])

K = V.shape[1]

print(f"Loaded model: K={K}, {len(movie_map)} movies, "
      f"{len(recommendable_codes)} recommendable (>={MIN_RATINGS_TO_RECOMMEND} ratings)")


# ===================== genre -> watch-time mapping =====================

GENRE_WATCH_SUGGESTIONS = {
    "Action":      ("Saturday afternoon",  "Grab some popcorn and buckle up for an action-packed Saturday matinee."),
    "Adventure":   ("Sunday morning",      "Start your Sunday with an epic adventure — pair it with a big breakfast."),
    "Animation":   ("Saturday morning",    "Relive Saturday-morning cartoon vibes with a bowl of cereal."),
    "Children":    ("Saturday morning",    "Perfect for a lazy Saturday morning with the whole family."),
    "Comedy":      ("Friday night",        "Kick off your weekend with laughs on Friday night."),
    "Crime":       ("Wednesday night",     "Break up the midweek grind with a gripping crime thriller."),
    "Documentary": ("Sunday evening",      "Wind down your weekend with something thought-provoking on Sunday evening."),
    "Drama":       ("Thursday night",      "Settle in on Thursday night for a powerful, character-driven story."),
    "Fantasy":     ("Sunday afternoon",    "Escape into another world on a lazy Sunday afternoon."),
    "Film-Noir":   ("Tuesday night",       "Set the mood on a quiet Tuesday night — dim the lights for this one."),
    "Horror":      ("Friday night",        "Friday the 13th energy — best watched late on a Friday night."),
    "Musical":     ("Sunday afternoon",    "Brighten your Sunday afternoon with show tunes and spectacle."),
    "Mystery":     ("Wednesday night",     "A midweek mystery is the perfect way to keep your mind sharp."),
    "Romance":     ("Saturday night",      "Date night special — pour some wine and enjoy a Saturday evening in."),
    "Sci-Fi":      ("Thursday night",      "Geek out on Thursday night with some mind-bending science fiction."),
    "Thriller":    ("Monday night",        "Beat the Monday blues with an edge-of-your-seat thriller."),
    "War":         ("Sunday evening",      "A reflective Sunday evening pairs well with a powerful war film."),
    "Western":     ("Saturday afternoon",  "Channel your inner cowboy on a Saturday afternoon."),
    "IMAX":        ("Saturday afternoon",  "Go big on Saturday afternoon — this one deserves the largest screen you have."),
    "(no genres listed)": ("Any evening", "Watch whenever the mood strikes."),
}

DEFAULT_SUGGESTION = ("Any evening", "A great pick for whenever you're in the mood.")


def get_watch_suggestion(genres_str):
    """Pick the best watch-time suggestion based on the first matching genre."""
    if not genres_str or pd.isna(genres_str):
        return DEFAULT_SUGGESTION

    for genre in genres_str.split("|"):
        genre = genre.strip()
        if genre in GENRE_WATCH_SUGGESTIONS:
            return GENRE_WATCH_SUGGESTIONS[genre]

    return DEFAULT_SUGGESTION


# ===================== recommendation logic =====================

def recommend_for_new_user(user_ratings: dict, n: int = 20):
    """
    user_ratings: { movieId (int) : rating (float) }
    Solves for a new user vector with adaptive regularisation,
    then scores only recommendable movies.
    """
    codes = []
    ratings = []
    for mid, r in user_ratings.items():
        mid = int(mid)
        if mid in movie_id_to_code:
            codes.append(movie_id_to_code[mid])
            ratings.append(float(r))

    if len(codes) == 0:
        return []

    codes_arr = np.array(codes)
    ratings_arr = np.array(ratings)

    V_u = V[codes_arr, :]                                # (n_rated, K)
    I_mat = np.eye(K)

    # Adaptive regularisation: stronger when fewer ratings
    # With 5 ratings use ~1.0, with 50 ratings use ~0.1
    lambda_new_user = max(0.1, 5.0 / len(codes))

    A = V_u.T @ V_u + lambda_new_user * I_mat            # (K, K)
    b = V_u.T @ ratings_arr                               # (K,)
    u_new = np.linalg.solve(A, b)                         # (K,)

    # Score only recommendable movies (pre-filtered for popularity)
    predictions = V_recommendable @ u_new                 # (n_recommendable,)

    # Clamp to valid rating range
    predictions = np.clip(predictions, 0.5, 5.0)

    # Exclude movies the user already rated
    rated_mids = set(int(mid) for mid in user_ratings.keys())
    scored = [
        (int(mid), float(pred))
        for mid, pred in zip(recommendable_mids, predictions)
        if int(mid) not in rated_mids
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:n]


# ===================== Flask app =====================

app = Flask(__name__)
CORS(app)


@app.route("/api/movies/popular", methods=["GET"])
def popular():
    n = request.args.get("n", 50, type=int)
    top = rating_frequency.head(n)
    result = top.to_dict(orient="records")
    return jsonify(result)


@app.route("/api/movies/search", methods=["GET"])
def search():
    q = request.args.get("q", "").strip().lower()
    if not q:
        return jsonify([])
    mask = movies_df['title'].str.lower().str.contains(q, na=False)
    matches = movies_df[mask].head(50)
    matches = matches.merge(
        rating_frequency[['movieId', 'count']],
        on='movieId', how='left'
    )
    matches['count'] = matches['count'].fillna(0).astype(int)
    result = matches.to_dict(orient="records")
    return jsonify(result)


@app.route("/api/recommend", methods=["POST"])
def recommend():
    body = request.get_json(force=True)
    user_ratings = body.get("ratings", {})
    n = body.get("n", 20)

    recs = recommend_for_new_user(user_ratings, n=n)

    rec_ids = [r[0] for r in recs]
    rec_scores = {r[0]: r[1] for r in recs}
    rec_df = movies_df[movies_df['movieId'].isin(rec_ids)].copy()
    rec_df['predicted_rating'] = rec_df['movieId'].map(rec_scores)
    rec_df = rec_df.sort_values('predicted_rating', ascending=False)

    # Attach watch suggestion
    results = []
    for _, row in rec_df.iterrows():
        when, blurb = get_watch_suggestion(row['genres'])
        results.append({
            "movieId": int(row['movieId']),
            "title": row['title'],
            "genres": row['genres'],
            "predicted_rating": round(float(row['predicted_rating']), 2),
            "watch_when": when,
            "watch_blurb": blurb,
        })

    return jsonify(results)


# Serve React static build in production
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve_frontend(path):
        file_path = os.path.join(static_dir, path)
        if path and os.path.exists(file_path):
            return send_from_directory(static_dir, path)
        return send_from_directory(static_dir, "index.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)