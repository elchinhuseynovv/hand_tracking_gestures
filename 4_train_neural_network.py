import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
import joblib
import json
import os

DATA_FILE = "data/az_data.csv"
MODEL_TF = "models/az_nn_model"
MODEL_TFLITE = "models/az_model.tflite"
LABEL_FILE = "models/az_labels.json"

print("Loading data...")
df = pd.read.csv(DATA_FILE, header = None)
X = df.iloc[:, :-1].values.astype(np.float32)
y = df.iloc[:, -1].values

encoder = LabelEncoder()
y_enc = encoder.fit_transform(y)
n_classes = len(encoder.classes_)

print(f"Total samples : {len(X)}")
print(f"Label found: {list(encoder.classes_)}")
print(f"Num classes : {n_classes}")

os.makedirs("models", exist_ok=True)
label_map = {i: label for i, label in enumerate(encoder.classes_)}
with open(LABEL_FILE, "w") as f:
    json.dump(label_map, f)
print(f"Labels saved : {LABEL_FILE}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
print(f"\nTraining samples : {len(X_train)}")
print(f"Testing samples : {len(X_test)}")

y_train_oh = tf.keras.utils.to_categorical(y_train, n_classes)
y_test_oh = tf.keras.utils.to_categorical(y_test, n_classes)

# Model ;)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(63,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(n_classes, activation='softmax')
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=10,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=5, min_lr=1e-6
    )
]

print("\nTraining neural network...")
history = model.fit(
    X_train, y_train_oh,
    validation_split=0.15,
    epochs = 100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

y_pred = model.predict(X_test)
y_pred_cls = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(y_test, y_pred_cls)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print("\nPer-letter breakdown:")
print(classification_report(
    y_test, y_pred_cls,
    target_names=encoder.classes_
))

model.save(MODEL_TF)
print(f"\nSaveModel saved to: {MODEL_TF}")

print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_TF)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(MODEL_TFLITE, "wb") as f:
    f.write(tflite_model)

size_kb = os.path.getsize(MODEL_TFLITE) / 1024
print(f"TFLite model saved to : {MODEL_TFLITE}")
print(f"TFLite model size : {size_kb:.1f} KB")
print("\nDone! Ready for mobile.")