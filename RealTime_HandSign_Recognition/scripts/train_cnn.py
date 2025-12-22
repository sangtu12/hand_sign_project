import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# ===============================
# PATH DATASET
# ===============================
train_dir = "dataset/images/train"
os.makedirs("model", exist_ok=True)

# ===============================
# PARAMETER
# ===============================
IMG_SIZE = (64, 64)
BATCH_SIZE = 16
EPOCHS = 10

# ===============================
# DATA GENERATOR
# ===============================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

NUM_CLASSES = train_gen.num_classes

print("Train samples:", train_gen.samples)
print("Val samples  :", val_gen.samples)
print("Classes      :", train_gen.class_indices)

# ===============================
# MODEL CNN
# ===============================
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(64,64,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ===============================
# TRAIN
# ===============================
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# ===============================
# SAVE MODEL
# ===============================
model.save("model/sign_language_cnn.h5")
print("✅ Model CNN berhasil disimpan")
