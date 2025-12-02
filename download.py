import pandas as pd

# Load the raw data
df = pd.read_csv("data/dataset.csv")

print(df.shape)          
print(df['Disease'].nunique()) 
print(df.columns.tolist())

# Symptoms are in columns like 'itching', 'skin_rash', ..., 'prognosis'
# 'prognosis' is the old name for Disease in some versions
if 'prognosis' in df.columns:
    df = df.rename(columns={'prognosis': 'Disease'})

# Remove the underscore spacing that some versions have
symptom_cols = [col for col in df.columns if col != 'Disease']
for col in symptom_cols:
    df[col] = df[col].astype(int)  # 1 = present, 0 = absent

print(df['Disease'].value_counts())