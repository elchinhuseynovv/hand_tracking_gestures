import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

DATA_FILE = "data/az_data.csv"
MODEL_FILE = "models/az_model.pkl"

print("Loading data and model...")
df = pd.read_csv(DATA_FILE, header=None)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
model = joblib.load(MODEL_FILE)

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

y_pred = model.predict(X_test)
labels = sorted(set(y))

# only for waek letters
weak = ['C', 'P']
for letter in weak:
    mask = y_test == letter
    preds = y_pred[mask]
    total = len(preds)
    correct = (preds == letter).sum()
    wrong = preds[preds != letter]

    print(f"\n── {letter} ──────────────────")
    print(f"Correct: {correct}/{total} ({correct/total*100:.1f}%)")
    if len(wrong) > 0:
        from collections import Counter
        confused = Counter(wrong).most_common(5)
        print("Most confused with:")
        for label, count in confused:
            print(f"  {label}: {count} times")