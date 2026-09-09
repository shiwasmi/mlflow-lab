from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
import numpy as np

# Point to MLflow tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# FastAPI app title
app = FastAPI(title="Digit Model API")

# Load Production model from MLflow Registry
model = mlflow.sklearn.load_model("models:/DigitModel/Production")

# Digits dataset has 10 classes (0–9)
class_names = [str(i) for i in range(10)]

# Request schema
class DigitRequest(BaseModel):
    features: list[float]

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(request: DigitRequest):
    X = np.array([request.features])
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    return {
        "predicted_class": class_names[int(prediction)],
        "probabilities": probabilities.tolist()
    }
