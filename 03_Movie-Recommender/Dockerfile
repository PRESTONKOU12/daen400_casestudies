# ============ Stage 1: Build React frontend ============
FROM node:20-alpine AS frontend-build

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ============ Stage 2: Python app ============
FROM python:3.11-slim

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source + data
COPY train-als.py recommend.py ./
COPY data/ data/

# Copy built React app into a static folder Flask will serve
COPY --from=frontend-build /app/frontend/dist /app/static

EXPOSE 5001

# Entrypoint: train the model, then start Flask
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh
CMD ["./entrypoint.sh"]