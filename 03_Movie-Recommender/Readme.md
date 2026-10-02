# Movie Recommender System

**DAEN 400 — Case Study 3**
Texas A&M University · Fall 2026

A full-stack movie recommender built from scratch using Alternating Least Squares (ALS) matrix factorisation. The system learns latent user and movie representations from the [MovieLens](https://grouplens.org/datasets/movielens/) dataset and generates personalised recommendations for new users in real time through an interactive web interface.

---

## Problem

Given a sparse user–movie rating matrix where ~98% of entries are unobserved, predict how a user would rate movies they haven't seen and surface the highest-scoring ones as recommendations.

Classical SVD requires a fully observed matrix, making it impractical without imputation. This project uses **ALS**, which operates directly on observed ratings by alternating between two closed-form least-squares solves:

1. Fix movie matrix **V**, solve for each user vector in **U**
2. Fix user matrix **U**, solve for each movie vector in **V**

At inference time, a new user rates a small subset of movies and the same closed-form update solves for their latent vector, enabling immediate recommendations without retraining.

---

## Architecture

| Component | Description |
|---|---|
| `train-als.py` | Trains U and V matrices via ALS, saves artifacts and RMSE convergence plot |
| `recommend.py` | Flask API — serves popular movies, search, and recommendations; serves the React build in production |
| `frontend/` | Vite + React app — movie browsing, star ratings, recommendation display with "when to watch" suggestions |
| `Dockerfile` | Multi-stage build (Node for frontend, Python for backend) |
| `docker-compose.yml` | One-command orchestration with volume-persisted artifacts |

---

## Quickstart (Docker — Recommended)

> **Prerequisites:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

```bash
# 1. Clone the repo
git clone <repo-url>
cd 03_Movie-Recommender

# 2. Build and run (first run trains the model — ~7 minutes)
docker compose up --build

# 3. Open in browser
# http://127.0.0.1:5001/
```
That's it. The container will:

1. Build the React frontend
2. Install Python dependencies
3. Train the ALS model (only on first run — artifacts are persisted)
4. Start the Flask server

docker compose down -v        # removes persisted artifacts
docker compose up --build


## Quickstart (Local Development)
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Train the model
python train-als.py

# 4. Start the Flask API (Terminal 1)
python recommend.py           # → http://localhost:5001

# 5. Start the React dev server (Terminal 2)
cd frontend
npm install
npm run dev                   # → http://localhost:3000 (proxies /api to Flask)