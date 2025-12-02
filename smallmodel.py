# make_final_model.py  ← THIS ONE IS BULLETPROOF

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import json
import os

os.makedirs("tiny_model", exist_ok=True)

df = pd.read_csv("data/dataset_clean_binary.csv")
df.rename(columns={'prognosis': 'Disease'}, inplace=True)
df.columns = df.columns.str.strip()

symptom_cols = [c for c in df.columns if c != 'Disease']
print(f"Loaded {len(df)} rows, {len(symptom_cols)} symptoms")
print("Sample row sum:", df[symptom_cols].iloc[0].sum())  # should be 3–8

X = df[symptom_cols].values
y = df['Disease'].values

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_val, y_train, y_val = train_test_split(X, y_enc, test_size=0.15, random_state=42, stratify=y_enc)

model = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

acc = (model.predict(X_val) == y_val).mean()
print(f"FINAL ACCURACY: {acc*100:.2f}%")

joblib.dump(model, "tiny_model/model.pkl")
joblib.dump(le, "tiny_model/label_encoder.pkl")
json.dump(symptom_cols, open("tiny_model/symptoms_list.json", "w"))

print("MODEL READY — deploy now!")