import joblib
import pandas as pd

MODEL_PATH = "models/heart_disease_model.joblib"

def load_model():
    return joblib.load(MODEL_PATH)

def predict_one(payload):
    model = load_model()
    X = pd.DataFrame([payload])
    prediction = int(model.predict(X)[0])
    confidence = float(model.predict_proba(X).max())
    return prediction, confidence
