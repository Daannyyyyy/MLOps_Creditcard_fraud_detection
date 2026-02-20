import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

from src.utils import preprocess_data

app = FastAPI(title="Credit card fraud detection", description="Checks whether a transaction is fraud", version = "1.0")
model = joblib.load("models/fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")

THRESHOLD = 0.9

class Transaction(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


@app.get("/")
def home():
    return {"Message:":"Credit card API is working"}

@app.post("/predict")
def predict(transaction:Transaction):
    data = pd.DataFrame([transaction.dict()])

    X = preprocess_data(data, scaler=scaler, training=False)
    prob = model.predict_proba(X)[:,1][0]
    prediction = int(prob>THRESHOLD)
    
    return {
        "fraud_probability": round(float(prob), 4),
        "prediction": prediction,
        "label": "Fraud" if prediction == 1 else "Not Fraud"
    }
