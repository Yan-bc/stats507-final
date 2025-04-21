# Position-Specific Football Player Analysis

This repository contains the full codebase and data for analyzing position-specific feature importance of football players using Random Forest, XGBoost, and SHAP on a FIFA dataset of 17,954 players.

## Repository Structure

```
stats507-final/
├── data/
│   ├── raw/                # Original CSV files
│   │   └── fifa_players.csv
│   └── processed/          # Cleaned & standardized data
│       └── fifa_players_cleaned.csv
├── src/
│   ├── preprocessing.py    # Data cleaning & normalization
│   ├── modeling.py         # Model training & SHAP analysis
│   └── visualization.py    # Radar plot generation
├── models/                 # Trained model files & feature JSONs
│   └── top5_features.json
├── figures/
│   ├── radar/              # Radar plots of top-5 features
│   │   └── radar_top5.png
│   └── shap/               # SHAP summary plots per position
│       ├── shap_FWD.png
│       ├── shap_MID.png
│       ├── shap_DEF.png
│       └── shap_GK.png
├── requirements.txt        # Pip dependencies
├── environment.yml         # Conda environment spec
└── run_all.sh              # One‑step pipeline script
```

## Setup

### Option A: Conda

1. Install Miniforge or Anaconda.  
2. Create the environment and activate it:
   ```bash
   conda env create -f environment.yml
   conda activate stats507
   ```

### Option B: Pip

1. Ensure you have Python 3.10+ installed.  
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

From the repository root:

```bash
# 1. Preprocess raw data
python3 src/preprocessing.py

# 2. Train models and compute SHAP values
python3 src/modeling.py

# 3. Generate radar plot of top-5 features
python3 src/visualization.py
```

Or run everything in one go:

```bash
bash run_all.sh
```

## Outputs

- **data/processed/fifa_players_cleaned.csv**: cleaned and scaled dataset  
- **models/**: Random Forest & XGBoost model files and `top5_features.json`  
- **figures/radar/**: radar_top5.png  
- **figures/shap/**: shap summary plots for FWD, MID, DEF, GK  

## Contact

Bicheng Yan  
University of Michigan  
Email: yanbc@umich.edu
