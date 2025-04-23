# src/modeling.py

import os
import json
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

if __name__ == '__main__':
    # Locate the project root directory
    base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_csv   = os.path.join(base_dir, 'data', 'processed', 'fifa_players_cleaned.csv')
    models_dir  = os.path.join(base_dir, 'models')
    shap_dir    = os.path.join(base_dir, 'figures', 'shap')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(shap_dir,   exist_ok=True)

    # Load the cleaned and normalized data
    df = load_data(input_csv)

    # Automatically identify numeric columns and exclude non-skill columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {
        'overall_rating',
        'age', 'height_cm', 'weight_kgs',
        'potential', 'value_euro', 'wage_euro',
        'release_clause_euro', 'national_rating',
        'national_jersey_number'
    }
    skill_cols = [c for c in num_cols if c not in exclude]
    print("Skill feature columns:", skill_cols)

    # Group by role, train models, and perform SHAP analysis
    top5_features = {}
    for role in df['role'].unique():
        sub = df[df['role'] == role].copy()
        X   = sub[skill_cols]
        y   = sub['overall_rating']

        print(f"\n>>> Role: {role} | Samples: {len(sub)}")

        # Baseline model: Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        r2_rf = cross_val_score(rf, X, y, cv=5, scoring='r2').mean()
        print(f"RF 5-fold CV R²: {r2_rf:.4f}")
        rf.fit(X, y)
        joblib.dump(rf, os.path.join(models_dir, f'rf_{role}.pkl'))

        # Enhanced model: XGBoost + GridSearch
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
        param_grid = {
            'max_depth': [4, 6],
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1]
        }
        grid = GridSearchCV(xgb_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
        grid.fit(X, y)
        best_xgb = grid.best_estimator_
        print(f"XGB best R²: {grid.best_score_:.4f} | Params: {grid.best_params_}")
        joblib.dump(best_xgb, os.path.join(models_dir, f'xgb_{role}.pkl'))

        # SHAP explanation
        explainer = shap.Explainer(best_xgb, X)
        shap_values = explainer(X)

        # Plot and save SHAP summary and bar charts
        shap.summary_plot(shap_values, X, show=False)
        shap.plots.bar(shap_values, max_display=10, show=False)
        import matplotlib.pyplot as plt
        plt.tight_layout()
        plt.savefig(os.path.join(shap_dir, f'shap_{role}.png'), bbox_inches='tight')
        plt.clf()

        # Extract Top-5 features
        mean_abs = np.abs(shap_values.values).mean(axis=0)
        top5 = list(pd.Series(mean_abs, index=skill_cols).nlargest(5).index)
        top5_features[role] = top5
        print(f"{role} Top-5 features: {top5}")

    # Save all roles' Top-5 features to JSON
    with open(os.path.join(models_dir, 'top5_features.json'), 'w') as f:
        json.dump(top5_features, f, indent=2, ensure_ascii=False)
    print("\nAll roles' Top-5 features saved:", top5_features)