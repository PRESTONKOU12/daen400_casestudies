"""
Movie recommender: biased matrix factorization trained with ALS (NumPy only).

Model:  r_hat(u, i) = mu + b_u + b_i + p_u . q_i

Pipeline:
  1. Load ratings.csv / movies.csv / links.csv (MovieLens column names).
  2. Per-user temporal split (last 20% of each user's ratings -> test).
  3. Optional hyperparameter search on a validation split carved from train.
  4. Evaluate RMSE + precision/recall@K vs. a bias-only baseline.
  5. Refit on ALL data and export JSON for the JavaScript UI.

Usage:
  python train_recommender.py --data-dir ./ml-latest-small --out-dir ./export
  python train_recommender.py --data-dir ./ml-latest-small --tune
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DATA_DIR = Path("data")       # folder containing ratings.csv, movies.csv, links.csv
OUT_DIR = Path("export")      # where the JSON files for the UI are written
N_FACTORS = 32                # latent dimension k
REG = 0.1                     # regularization strength
N_ITERS = 15                  # ALS iterations
TUNE = False                  # True -> grid-search N_FACTORS/REG on a validation split


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_data(data_dir: Path):
    ratings = pd.read_csv(data_dir / "ratings.csv")   # userId, movieId, rating, timestamp
    movies = pd.read_csv(data_dir / "movies.csv")     # movieId, title, genres
    links = pd.read_csv(data_dir / "links.csv")       # movieId, imdbId, tmdbId
    required = {"userId", "movieId", "rating", "timestamp"}
    missing = required - set(ratings.columns)
    if missing:
        raise ValueError(f"ratings.csv is missing columns: {missing}")
    return ratings, movies, links


def temporal_split(ratings: pd.DataFrame, test_frac=0.2, min_ratings=5):
    """Hold out each user's most recent `test_frac` of ratings.
    Users with fewer than `min_ratings` ratings stay entirely in train."""
    df = ratings.sort_values(["userId", "timestamp"], kind="stable")
    rank = df.groupby("userId").cumcount()
    n = df.groupby("userId")["movieId"].transform("size")
    cutoff = np.floor(n * (1 - test_frac)).astype(int)
    is_test = (rank >= cutoff) & (n >= min_ratings)
    return df[~is_test].copy(), df[is_test].copy()


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class BiasedALS:
    def __init__(self, n_factors=32, reg=0.1, reg_bias=0.1, n_iters=15,
                 min_item_ratings=10, seed=0, verbose=True):
        self.k = n_factors
        self.reg = reg              # weighted-lambda (ALS-WR): scaled by #ratings per row
        self.reg_bias = reg_bias
        self.n_iters = n_iters
        self.min_item_ratings = min_item_ratings  # items eligible to be *recommended*
        self.seed = seed
        self.verbose = verbose

    # -- helpers --------------------------------------------------------------
    @staticmethod
    def _group(keys, others, vals, n):
        """Return list where entry j = (other indices, ratings) for key j."""
        order = np.argsort(keys, kind="stable")
        keys, others, vals = keys[order], others[order], vals[order]
        bounds = np.searchsorted(keys, np.arange(n + 1))
        return [(others[bounds[j]:bounds[j + 1]], vals[bounds[j]:bounds[j + 1]])
                for j in range(n)]

    def _update(self, X, bx, Y, by, groups):
        """Solve for rows of X (and biases bx) with Y, by fixed.
        For row j: minimize sum (r - mu - by - [x_j, b_j].[y, 1])^2 + ridge."""
        k = self.k
        base_reg = np.diag([self.reg] * k + [self.reg_bias])
        for j, (cols, vals) in enumerate(groups):
            if len(cols) == 0:
                continue
            A = np.hstack([Y[cols], np.ones((len(cols), 1))])
            target = vals - self.mu - by[cols]
            lhs = A.T @ A + base_reg * len(cols)
            sol = np.linalg.solve(lhs, A.T @ target)
            X[j] = sol[:k]
            bx[j] = sol[k]

    # -- training -------------------------------------------------------------
    def fit(self, df: pd.DataFrame):
        self.user_ids = np.unique(df["userId"].to_numpy())
        self.item_ids = np.unique(df["movieId"].to_numpy())
        self.u_index = {int(u): j for j, u in enumerate(self.user_ids)}
        self.i_index = {int(m): j for j, m in enumerate(self.item_ids)}

        u = df["userId"].map(self.u_index).to_numpy()
        i = df["movieId"].map(self.i_index).to_numpy()
        r = df["rating"].to_numpy(dtype=float)
        nU, nI = len(self.user_ids), len(self.item_ids)

        self.mu = r.mean()
        rng = np.random.default_rng(self.seed)
        self.P = rng.normal(0, 0.1, (nU, self.k))
        self.Q = rng.normal(0, 0.1, (nI, self.k))
        self.bu = np.zeros(nU)
        self.bi = np.zeros(nI)

        item_counts = np.bincount(i, minlength=nI)
        self.eligible = item_counts >= self.min_item_ratings

        by_user = self._group(u, i, r, nU)
        by_item = self._group(i, u, r, nI)

        for it in range(self.n_iters):
            self._update(self.P, self.bu, self.Q, self.bi, by_user)
            self._update(self.Q, self.bi, self.P, self.bu, by_item)
            if self.verbose:
                pred = self.mu + self.bu[u] + self.bi[i] + np.sum(self.P[u] * self.Q[i], axis=1)
                print(f"  iter {it + 1:2d}  train RMSE {np.sqrt(np.mean((r - pred) ** 2)):.4f}")
        return self

    # -- inference ------------------------------------------------------------
    def predict(self, users, movies):
        uidx = np.array([self.u_index.get(int(x), -1) for x in users])
        iidx = np.array([self.i_index.get(int(x), -1) for x in movies])
        pred = np.full(len(uidx), self.mu)
        ku, ki = uidx >= 0, iidx >= 0
        pred[ku] += self.bu[uidx[ku]]           # unknown item -> mu + b_u
        pred[ki] += self.bi[iidx[ki]]           # unknown user -> mu + b_i
        both = ku & ki
        pred[both] += np.sum(self.P[uidx[both]] * self.Q[iidx[both]], axis=1)
        return np.clip(pred, 0.5, 5.0)

    def recommend(self, user_id, n=10, exclude=()):
        u = self.u_index[int(user_id)]
        scores = self.mu + self.bu[u] + self.bi + self.Q @ self.P[u]
        scores[~self.eligible] = -np.inf
        ex = [self.i_index[m] for m in exclude if m in self.i_index]
        scores[ex] = -np.inf
        n = min(n, int(np.isfinite(scores).sum()))
        top = np.argpartition(-scores, n - 1)[:n]
        top = top[np.argsort(-scores[top])]
        return [(int(self.item_ids[j]), float(min(scores[j], 5.0))) for j in top]

    def all_similar_items(self, n=10, chunk=1024):
        """Cosine similarity on item embeddings, restricted to eligible items."""
        idx = np.flatnonzero(self.eligible)
        Qe = self.Q[idx]
        Qe = Qe / (np.linalg.norm(Qe, axis=1, keepdims=True) + 1e-9)
        out = {}
        for s in range(0, len(idx), chunk):
            S = Qe[s:s + chunk] @ Qe.T
            for row, j in enumerate(range(s, min(s + chunk, len(idx)))):
                S[row, j] = -np.inf  # exclude self
            top = np.argpartition(-S, n, axis=1)[:, :n]
            for row in range(S.shape[0]):
                t = top[row][np.argsort(-S[row, top[row]])]
                out[int(self.item_ids[idx[s + row]])] = [
                    {"movieId": int(self.item_ids[idx[c]]), "sim": round(float(S[row, c]), 4)}
                    for c in t]
        return out


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def rmse(model, test):
    pred = model.predict(test["userId"].to_numpy(), test["movieId"].to_numpy())
    return float(np.sqrt(np.mean((test["rating"].to_numpy() - pred) ** 2)))


def ranking_metrics(model, train, test, k=10, like_threshold=4.0):
    """Precision/recall@k where 'relevant' = held-out rating >= like_threshold."""
    seen = train.groupby("userId")["movieId"].apply(set)
    relevant = test[test["rating"] >= like_threshold].groupby("userId")["movieId"].apply(set)
    precs, recs = [], []
    for uid, rel in relevant.items():
        if int(uid) not in model.u_index:
            continue
        top = {m for m, _ in model.recommend(uid, k, exclude=seen.get(uid, set()))}
        hits = len(rel & top)
        precs.append(hits / k)
        recs.append(hits / len(rel))
    return float(np.mean(precs)), float(np.mean(recs))


def tune(train, factors=(8, 16, 32, 64), regs=(0.02, 0.05, 0.1, 0.2), n_iters=10):
    """Grid search on a validation split carved from TRAIN (never touch test)."""
    tr, val = temporal_split(train, test_frac=0.2)
    best = None
    for k in factors:
        for lam in regs:
            m = BiasedALS(n_factors=k, reg=lam, reg_bias=lam, n_iters=n_iters,
                          verbose=False).fit(tr)
            score = rmse(m, val)
            print(f"  k={k:3d} reg={lam:<5}  val RMSE {score:.4f}")
            if best is None or score < best[0]:
                best = (score, k, lam)
    print(f"Best: k={best[1]}, reg={best[2]} (val RMSE {best[0]:.4f})")
    return best[1], best[2]


# ----------------------------------------------------------------------------
# Export for the JavaScript UI
# ----------------------------------------------------------------------------
def export(model, ratings, movies, links, out_dir: Path, n_recs=20, n_similar=10):
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = movies.merge(links, on="movieId", how="left")
    movie_json = {
        int(row.movieId): {
            "title": row.title,
            "genres": row.genres.split("|") if isinstance(row.genres, str) else [],
            "imdbId": None if pd.isna(row.imdbId) else f"tt{int(row.imdbId):07d}",
            "tmdbId": None if pd.isna(row.tmdbId) else int(row.tmdbId),
        }
        for row in meta.itertuples(index=False)
    }

    seen = ratings.groupby("userId")["movieId"].apply(set)
    recs_json = {
        int(uid): [{"movieId": m, "score": round(s, 3)}
                   for m, s in model.recommend(uid, n_recs, exclude=seen.get(uid, set()))]
        for uid in model.user_ids
    }

    similar_json = model.all_similar_items(n=n_similar)

    for name, obj in [("movies.json", movie_json),
                      ("recommendations.json", recs_json),
                      ("similar_movies.json", similar_json)]:
        with open(out_dir / name, "w") as f:
            json.dump(obj, f)
        print(f"  wrote {out_dir / name}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ratings, movies, links = load_data(DATA_DIR)
    n_u, n_i = ratings["userId"].nunique(), ratings["movieId"].nunique()
    print(f"{len(ratings):,} ratings | {n_u:,} users | {n_i:,} movies | "
          f"sparsity {1 - len(ratings) / (n_u * n_i):.2%}")

    train, test = temporal_split(ratings)
    print(f"Train {len(train):,} / Test {len(test):,} (per-user temporal split)")

    k, lam = (tune(train) if TUNE else (N_FACTORS, REG))

    print("\nBaseline (biases only, k=0):")
    base = BiasedALS(n_factors=0, reg=lam, reg_bias=lam, n_iters=N_ITERS,
                     verbose=False).fit(train)
    print(f"\nMatrix factorization (k={k}, reg={lam}):")
    mf = BiasedALS(n_factors=k, reg=lam, reg_bias=lam, n_iters=N_ITERS).fit(train)

    print("\n=== Test results ===")
    for name, m in [("Bias baseline", base), ("Biased MF (ALS)", mf)]:
        p, r = ranking_metrics(m, train, test)
        print(f"{name:18s} RMSE {rmse(m, test):.4f} | P@10 {p:.4f} | R@10 {r:.4f}")

    print("\nRefitting on all data for export...")
    final = BiasedALS(n_factors=k, reg=lam, reg_bias=lam, n_iters=N_ITERS,
                      verbose=False).fit(ratings)
    export(final, ratings, movies, links, OUT_DIR)


if __name__ == "__main__":
    main()