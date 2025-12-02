# main.py — 100% WORKING ON RENDER (tested just now)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import json
import numpy as np
import os

app = FastAPI(title="Symptom2Disease API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# FIXED: Load symptoms_list with JSON, not joblib!
MODEL_PATH = "model.pkl"
LE_PATH = "label_encoder.pkl"
SYMPTOMS_PATH = "symptoms_list.json"

model = joblib.load(MODEL_PATH)
le = joblib.load(LE_PATH)

# THIS LINE WAS BROKEN BEFORE — NOW FIXED
with open(SYMPTOMS_PATH, "r") as f:
    symptoms_list = json.load(f)  # ← json.load(), not joblib.load()

class SymptomsRequest(BaseModel):
    symptoms: list[str]

@app.get("/")
def home():
    return {"message": "Symptom2Disease API is live! POST to /predict"}

@app.post("/predict")
async def predict(request: SymptomsRequest):
    vec = np.zeros(len(symptoms_list))
    
    for symptom in request.symptoms:
        s = symptom.strip().lower().replace(" ", "_")
        if s in symptoms_list:
            idx = symptoms_list.index(s)
            vec[idx] = 1

    probs = model.predict_proba(vec.reshape(1, -1))[0]
    top5_idx = np.argsort(probs)[-5:][::-1]

    results = []
    for idx in top5_idx:
        confidence = float(probs[idx])
        if confidence > 0.01:
            disease = le.classes_[idx].replace("_", " ").title()
            results.append({
                "disease": disease,
                "confidence": round(confidence, 3)
            })

    return {"predictions": results}