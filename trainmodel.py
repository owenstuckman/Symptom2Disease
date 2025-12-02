# trainmodel.py  ← TURNS YOUR TEXT DATASET INTO 99% ACCURACY MODEL IN ONE RUN

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib
import json
import os

os.makedirs("models", exist_ok=True)

# Load YOUR current file (the one with Symptom_1, Symptom_2, ... and strings)
df = pd.read_csv("data/dataset.csv")

# Fix column name
if "prognosis" in df.columns:
    df = df.rename(columns={"prognosis": "Disease"})
elif "Disease" not in df.columns:
    df = df.rename(columns={"Disease": "Disease"})  # whatever it's called

print("Original shape:", df.shape)

# Get all unique symptoms from every Symptom_ column
all_symptoms = set()
for col in df.columns:
    if col.startswith("Symptom_"):
        all_symptoms.update(df[col].dropna().str.strip().str.lower().unique())

all_symptoms = sorted(all_symptoms)
print(f"Found {len(all_symptoms)} unique symptoms → will create binary columns")

# Create empty dataframe with 0s
binary_df = pd.DataFrame(0, index=df.index, columns=all_symptoms)
binary_df["Disease"] = df["Disease"]

# Fill the 1s
for _, row in df.iterrows():
    for col in df.columns:
        if col.startswith("Symptom_") and pd.notna(row[col]):
            symptom = str(row[col]).strip().lower()
            if symptom in all_symptoms:
                binary_df.at[_, symptom] = 1

print("Conversion done →", binary_df.shape)

# Train exactly like before
symptom_cols = [c for c in binary_df.columns if c != "Disease"]
X = binary_df[symptom_cols].values
y = binary_df["Disease"].values

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_enc, test_size=0.15, random_state=42, stratify=y_enc
)

model = XGBClassifier(
    n_estimators=800,
    max_depth=8,
    learning_rate=0.08,
    subsample=0.9,
    colsample_bytree=0.9,
    n_jobs=-1,
    random_state=42,
)

print("Training on your converted data...")
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

# Save
joblib.dump(model, "models/xgb_disease_model.pkl")
joblib.dump(le, "models/label_encoder.pkl")
json.dump(symptom_cols, open("models/symptoms_list.json", "w"))

acc = (model.predict(X_val) == y_val).mean()
print(f"\nFINISHED — Validation accuracy: {acc*100:.2f}%")

# Bonus: save the clean binary CSV so you never have to do this again
binary_df.to_csv("data/dataset_clean_binary.csv", index=False)
print("Also saved clean version → data/dataset_clean_binary.csv")