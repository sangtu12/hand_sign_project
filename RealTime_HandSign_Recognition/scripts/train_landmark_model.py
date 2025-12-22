import os
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_DIR = "dataset/landmarks"

X, y = [], []

for file in os.listdir(DATA_DIR):
    if not file.endswith(".csv"):
        continue

    label = file.replace(".csv", "")
    path = os.path.join(DATA_DIR, file)

    df = pd.read_csv(path, header=None)

    if df.empty:
        print(f"⚠️ File kosong dilewati: {file}")
        continue

    X.append(df.values)
    y.extend([label] * len(df))

X = np.vstack(X)

encoder = LabelEncoder()
y = encoder.fit_transform(y)
y = tf.keras.utils.to_categorical(y)

print("📌 Kelas terdeteksi:", encoder.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation="relu", input_shape=(63,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(y.shape[1], activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test)
)

os.makedirs("model", exist_ok=True)
model.save("model/hand_landmark_model.h5")
joblib.dump(encoder, "model/label_encoder.pkl")

print("✅ Model & LabelEncoder berhasil disimpan")
