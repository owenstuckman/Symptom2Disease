# main.py — FOR YOUR LARGE XGBOOST MODEL (model.pkl)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import json
import numpy as np

app = FastAPI(title="Symptom2Disease XGBoost API")

# Allow Flutter to call it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your large XGBoost model + files
model = joblib.load("model.pkl")                    # ← your big model
le = joblib.load("label_encoder.pkl")
with open("symptoms_list.json", "r") as f:
    symptoms_list = json.load(f)                    # ← fixed with json.load()

class SymptomsRequest(BaseModel):
    symptoms: list[str]

@app.get("/")
def home():
    return {"status": "XGBoost model loaded", "diseases": len(le.classes_)}

@app.post("/predict")
def predict(request: SymptomsRequest):
    # Create input vector
    vec = np.zeros(len(symptoms_list))
    for symptom in request.symptoms:
        s = symptom.strip().lower().replace(" ", "_")
        if s in symptoms_list:
            idx = symptoms_list.index(s)
            vec[idx] = 1

    # Predict probabilities
    probs = model.predict_proba(vec.reshape(1, -1))[0]
    top5_idx = probs.argsort()[-5:][::-1]

    results = []
    for idx in top5_idx:
        confidence = float(probs[idx])
        if confidence > 0.01:  # show only meaningful predictions
            disease = le.classes_[idx].replace("_", " ").title()
            results.append({
                "disease": disease,
                "confidence": round(confidence, 4)
            })

    return {"predictions": results}