# src/preprocessing.py

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_raw(path: str) -> pd.DataFrame:
    """Load raw CSV data from the given path."""
    return pd.read_csv(path)

def simplify_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Map the 'positions' column to four main role categories."""
    # Extract the primary position (e.g., from "RW,ST" take "RW")
    df['primary_position'] = df['positions'].str.split(',', expand=True)[0]

    # Mapping dictionary: forward, midfield, defense, goalkeeper
    mapping = {
        'ST':'FWD','CF':'FWD','CAM':'MID','CM':'MID','CDM':'MID',
        'LW':'FWD','RW':'FWD','LAM':'MID','RAM':'MID','LM':'MID','RM':'MID',
        'LB':'DEF','LWB':'DEF','CB':'DEF','RB':'DEF','RWB':'DEF',
        'GK':'GK'
    }
    # Apply mapping to create the 'role' column
    df['role'] = df['primary_position'].map(mapping)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and clip outliers."""
    # Drop rows without a mapped role
    df = df.dropna(subset=['role'])
    # Clip numeric columns to [1, 100] and fill missing with median
    numeric = df.select_dtypes(include=['int64','float64']).columns
    for col in numeric:
        df[col] = df[col].clip(lower=1, upper=100)
        df[col].fillna(df[col].median(), inplace=True)
    # Fill categorical columns with the mode
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def scale_numeric_features_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize only the numeric columns in the DataFrame,
    automatically ignoring all non-numeric columns.
    """
    # Identify all numeric columns
    num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    # Remove the target column if present
    if 'overall_rating' in num_cols:
        num_cols.remove('overall_rating')
    print("The following numeric features will be standardized:", num_cols)

    # Apply StandardScaler to numeric features
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

if __name__ == '__main__':
    base     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_csv  = os.path.join(base, 'data', 'raw', 'fifa_players.csv')
    proc_csv = os.path.join(base, 'data', 'processed', 'fifa_players_cleaned.csv')

    # Load, simplify positions, and clean the data
    df = load_raw(raw_csv)
    df = simplify_positions(df)
    df = clean_data(df)

    # Standardize numeric features only
    df = scale_numeric_features_only(df)

    # Save the processed data
    os.makedirs(os.path.dirname(proc_csv), exist_ok=True)
    df.to_csv(proc_csv, index=False)
    print(f"Processed data saved to: {proc_csv}")