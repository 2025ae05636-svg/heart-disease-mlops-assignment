from pathlib import Path

import pandas as pd

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
]

def load_data(path="data/raw/heart_disease.csv"):
    return pd.read_csv(path)

def clean_target(df):
    df = df.copy()
    # UCI Cleveland target: 0 = absence, 1-4 = presence.
    df["target"] = (df["target"] > 0).astype(int)
    return df

def prepare_xy(df):
    df = clean_target(df)
    X = df[FEATURES].copy()
    y = df["target"].copy()
    return X, y

def save_cleaned(df, path="data/processed/heart_disease_clean.csv"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    clean_target(df).to_csv(path, index=False)
