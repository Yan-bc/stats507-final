#!/usr/bin/env bash
set -e

echo ">> Step 1: Preprocessing data"
python3 src/preprocessing.py

echo ">> Step 2: Modeling & SHAP analysis"
python3 src/modeling.py

echo ">> Step 3: Visualization"
python3 src/visualization.py

echo "All done! Check data/processed, models/, figures/"
