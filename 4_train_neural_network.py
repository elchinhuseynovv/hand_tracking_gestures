import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import os

DATA_FILE    = "data/az_data.csv"
MODEL_PT     = "models/az_nn_model.pt"
MODEL_TFLITE = "models/az_model.ptl"
LABEL_FILE   = "models/az_labels.json"
EPOCHS       = 100
BATCH_SIZE   = 32
LR           = 0.001
PATIENCE     = 10

print("Loading data...")
df    = pd.read_csv(DATA_FILE, header=None)
X     = df.iloc[:, :-1].values.astype(np.float32)
y     = df.iloc[:, -1].values

encoder   = LabelEncoder()
y_enc     = encoder.fit_transform(y)
n_classes = len(encoder.classes_)

print(f"Total samples : {len(X)}")
print(f"Labels found  : {list(encoder.classes_)}")
print(f"Num classes   : {n_classes}")

os.makedirs("models", exist_ok=True)
label_map = {str(i): label for i, label in enumerate(encoder.classes_)}
with open(LABEL_FILE, "w", encoding="utf-8") as f:
    json.dump(label_map, f, ensure_ascii=False)
print(f"Labels saved  : {LABEL_FILE}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
print(f"\nTraining samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t  = torch.tensor(X_test)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

train_ds     = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# Model
class SignNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(63, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.net(x)

model     = SignNet(n_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.5, patience=5
)

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

# train
print("\nTraining neural network...")
best_acc    = 0.0
patience_ct = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        out  = model(X_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    with torch.no_grad():
        val_out  = model(X_test_t)
        val_pred = torch.argmax(val_out, dim=1)
        val_acc  = (val_pred == y_test_t).float().mean().item()
        val_loss = criterion(val_out, y_test_t).item()

    scheduler.step(val_loss)

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{EPOCHS} "
              f"| Loss: {total_loss/len(train_loader):.4f} "
              f"| Val Acc: {val_acc*100:.2f}%")

    if val_acc > best_acc:
        best_acc    = val_acc
        patience_ct = 0
        torch.save(model.state_dict(), MODEL_PT)
    else:
        patience_ct += 1
        if patience_ct >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

model.load_state_dict(torch.load(MODEL_PT))
model.eval()
with torch.no_grad():
    y_pred_t   = torch.argmax(model(X_test_t), dim=1).numpy()

accuracy = accuracy_score(y_test, y_pred_t)
print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%")
print("\nPer-letter breakdown:")
print(classification_report(y_test, y_pred_t, target_names=encoder.classes_))

print("Exporting to TorchScript for mobile...")
model.eval()
example   = torch.rand(1, 63)
scripted  = torch.jit.trace(model, example)
scripted.save(MODEL_TFLITE)

size_kb = os.path.getsize(MODEL_TFLITE) / 1024
print(f"\nModel saved to : {MODEL_PT}")
print(f"Mobile model   : {MODEL_TFLITE}  ({size_kb:.1f} KB)")
print("\nDone! Ready for mobile.")