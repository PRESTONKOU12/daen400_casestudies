#!/bin/bash
set -e

# Only train if artifacts are missing
if [ ! -f artifacts/User_Matrix.npy ]; then
    echo ">>> No artifacts found. Training model..."
    python train-als.py
else
    echo ">>> Artifacts found. Skipping training."
    echo "    (Delete artifacts/ and rebuild to retrain)"
fi

echo ">>> Starting Flask server..."
python recommend.py