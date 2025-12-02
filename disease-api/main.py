# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="Disease Predictor API", version="1.0")

# Allow Flutter/web to call it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained model (from tiny_model folder or root)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
LE_PATH = os.path.join(os.path.dirname(__file__), "label_encoder.pkl")
SYMPTOMS_PATH = os.path.join(os.path.dirname(__file__), "symptoms_list.json")

model = joblib.load(MODEL_PATH)
le = joblib.load(LE_PATH)
with open(SYMPTOMS_PATH) as f:
    symptoms_list = joblib.load(f)  # it's a list

class SymptomsRequest(BaseModel):
    symptoms: list[str]

@app.post("/predict")
async def predict(request: SymptomsRequest):
    # Create zero vector
    vec = np.zeros(len(symptoms_list))
    
    for symptom in request.symptoms:
        s = symptom.strip().lower().replace(" ", "_")
        if s in symptoms_list:
            idx = symptoms_list.index(s)
            vec[idx] = 1

    # Predict
    probs = model.predict_proba(vec.reshape(1, -1))[0]
    top5_idx = np.argsort(probs)[-5:][::-1]

    results = []
    for idx in top5_idx:
        if probs[idx] > 0.01:  # confidence > 1%
            disease = le.classes_[idx].replace("_", " ").title()
            results.append({
                "disease": disease,
                "confidence": round(float(probs[idx]), 3)
            })

    return {"predictions": results}