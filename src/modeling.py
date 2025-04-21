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
    # 定位项目根目录
    base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_csv   = os.path.join(base_dir, 'data', 'processed', 'fifa_players_cleaned.csv')
    models_dir  = os.path.join(base_dir, 'models')
    shap_dir    = os.path.join(base_dir, 'figures', 'shap')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(shap_dir,   exist_ok=True)

    # 1. 读取清洗并标准化后的数据
    df = load_data(input_csv)

    # 2. 自动识别数值列，并排除非“技能”列
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {
        'overall_rating',
        'age', 'height_cm', 'weight_kgs',
        'potential', 'value_euro', 'wage_euro',
        'release_clause_euro', 'national_rating',
        'national_jersey_number'
    }
    skill_cols = [c for c in num_cols if c not in exclude]
    print("技能特征列：", skill_cols)

    # 3. 按角色分组，训练模型并做 SHAP 分析
    top5_features = {}
    for role in df['role'].unique():
        sub = df[df['role'] == role].copy()
        X   = sub[skill_cols]
        y   = sub['overall_rating']

        print(f"\n>>> Role: {role} | 样本量: {len(sub)}")

        # 3.1 基线模型：Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        r2_rf = cross_val_score(rf, X, y, cv=5, scoring='r2').mean()
        print(f"RF 5-fold CV R²: {r2_rf:.4f}")
        rf.fit(X, y)
        joblib.dump(rf, os.path.join(models_dir, f'rf_{role}.pkl'))

        # 3.2 强化模型：XGBoost + GridSearch
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
        param_grid = {
            'max_depth': [4, 6],
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1]
        }
        grid = GridSearchCV(xgb_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
        grid.fit(X, y)
        best_xgb = grid.best_estimator_
        print(f"XGB 最优 R²: {grid.best_score_:.4f} | 参数: {grid.best_params_}")
        joblib.dump(best_xgb, os.path.join(models_dir, f'xgb_{role}.pkl'))

        # 3.3 SHAP 解释
        explainer = shap.Explainer(best_xgb, X)
        shap_values = explainer(X)

        # 绘制并保存 SHAP summary 和 bar 图
        shap.summary_plot(shap_values, X, show=False)
        shap.plots.bar(shap_values, max_display=10, show=False)
        import matplotlib.pyplot as plt
        plt.tight_layout()
        plt.savefig(os.path.join(shap_dir, f'shap_{role}.png'), bbox_inches='tight')
        plt.clf()

        # 3.4 提取 Top‑5 特征
        mean_abs = np.abs(shap_values.values).mean(axis=0)
        top5 = list(pd.Series(mean_abs, index=skill_cols).nlargest(5).index)
        top5_features[role] = top5
        print(f"{role} Top‑5 特征：{top5}")

    # 4. 保存所有角色的 Top‑5 特征到 JSON
    with open(os.path.join(models_dir, 'top5_features.json'), 'w') as f:
        json.dump(top5_features, f, indent=2, ensure_ascii=False)
    print("\n所有角色 Top‑5 特征已保存：", top5_features)
