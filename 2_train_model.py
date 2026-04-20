import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# CONFIG

DATA_FILE   = "data/asl_data.csv"
MODEL_FILE  = "models/asl_model.pkl"

print("Loading data...")
df = pd.read_csv(DATA_FILE, header=None)

x = df.iloc[:, :-1].values # 63 features
y = df.iloc[:, -1].values  # label (A-Z)

print(f"Total samples: {len(X)}")
print(f"Labels found: {sorted(set(y))}")
