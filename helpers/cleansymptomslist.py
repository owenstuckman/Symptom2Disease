# fix_symptoms_list.py  ← RUN THIS EXACTLY

import json
import joblib
import pandas as pd

# Load your original dataset to get the exact column order
df = pd.read_csv("data/dataset.csv")  # ← make sure this file exists!

# Rename if needed
if 'prognosis' in df.columns:
    df = df.rename(columns={'prognosis': 'Disease'})

# Strip any whitespace from column names (this was causing bugs before)
df.columns = df.columns.str.strip()

# Get only symptom columns (exclude Disease)
symptom_columns = [col for col in df.columns if col != 'Disease']

print(f"Found {len(symptom_columns)} symptoms")
print("First 10:", symptom_columns[:10])

# Save as proper JSON (not joblib!)
with open("symptoms_list.json", "w") as f:
    json.dump(symptom_columns, f, indent=2)

print("\nSUCCESS! symptoms_list.json created correctly.")
print("Now upload this file + model.pkl + label_encoder.pkl to your Render repo.")