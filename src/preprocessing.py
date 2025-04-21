# src/preprocessing.py
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_raw(path: str) -> pd.DataFrame:
    """加载原始 CSV 数据"""
    return pd.read_csv(path)

def simplify_positions(df: pd.DataFrame) -> pd.DataFrame:
    """将 CSV 中的 positions 列映射为四大角色类别"""
    # 先取逗号前的“首选位置”，如 "RW,ST" ➝ "RW"
    df['primary_position'] = df['positions'].str.split(',', expand=True)[0]

    # 映射字典：前锋、前腰、中后腰、后卫、守门员 等
    mapping = {
        'ST':'FWD','CF':'FWD','CAM':'MID','CM':'MID','CDM':'MID',
        'LW':'FWD','RW':'FWD','LAM':'MID','RAM':'MID','LM':'MID','RM':'MID',
        'LB':'DEF','LWB':'DEF','CB':'DEF','RB':'DEF','RWB':'DEF',
        'GK':'GK'
    }
    # 映射 primary_position 到 role
    df['role'] = df['primary_position'].map(mapping)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """处理缺失值与异常值"""
    # 丢弃无法映射到 role 的行
    df = df.dropna(subset=['role'])
    # 数值列上下限剪枝并填中位数
    numeric = df.select_dtypes(include=['int64','float64']).columns
    for col in numeric:
        df[col] = df[col].clip(lower=1, upper=100)
        df[col].fillna(df[col].median(), inplace=True)
    # 分类列填众数
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def scale_numeric_features_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    只对 DataFrame 中的数值列做标准化，自动忽略所有非数值列。
    """
    # 1. 自动找出所有数值列
    num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    # （可选）如果你有目标列也在数值列里，需要把它移除：
    if 'overall_rating' in num_cols:
        num_cols.remove('overall_rating')
    print("将标准化以下数值特征：", num_cols)

    # 2. 标准化
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

if __name__ == '__main__':
    base       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_csv    = os.path.join(base, 'data', 'raw', 'fifa_players.csv')
    proc_csv   = os.path.join(base, 'data', 'processed', 'fifa_players_cleaned.csv')

    df = load_raw(raw_csv)
    df = simplify_positions(df)
    df = clean_data(df)

    # —— 这里用新函数替代原来的 scale_features/skill_cols 逻辑 —— 
    df = scale_numeric_features_only(df)

    os.makedirs(os.path.dirname(proc_csv), exist_ok=True)
    df.to_csv(proc_csv, index=False)
    print(f"Processed data saved to: {proc_csv}")