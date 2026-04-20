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

print(f"Total samples: {len(x)}")
print(f"Labels found: {sorted(set(y))}")


# split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:    {len(X_test)}")


# train
print("\nTraining model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=1              #uses all CPU cores
)
model.fit(X_train, y_train)

# evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("\nPer-letter breakdown:")
print(classification_report(y_test, y_pred))

# save
os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_FILE)
print(f"\nModel saved to {MODEL_FILE}")