from pathlib import Path

import pandas as pd

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

def download_data(output_path="data/raw/heart_disease.csv"):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(URL, header=None, names=COLUMNS, na_values="?")
    df.to_csv(output, index=False)
    return df

if __name__ == "__main__":
    df = download_data()
    print(f"Saved {len(df)} rows to data/raw/heart_disease.csv")
    print(df.head())
