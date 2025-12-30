import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import time
import os

# ===============================
# SETTINGS (OPTIMIZED)
# ===============================
IMG_SIZE = (64, 64)
BATCH_SIZE = 128      # ✅ Besar untuk efisiensi
EPOCHS = 20

train_dir = "dataset/images/train"
test_dir = "dataset/images/test"

print("\n" + "="*60)
print("🚀 OPTIMIZED TRAINING (No Preload)")
print("="*60)
print(f"Image: {IMG_SIZE} | Batch: {BATCH_SIZE} | Epochs: {EPOCHS}")
print("="*60 + "\n")

# ===============================
# DATA GENERATOR (MINIMAL AUGMENTATION)
# ===============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,     # ✅ Minimal augmentation
    width_shift_range=0.1,
    height_shift_range=0.1
)

test_datagen = ImageDataGenerator(rescale=1./255)

# ===============================
# FLOW FROM DIRECTORY
# ===============================
train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

NUM_CLASSES = train_gen.num_classes

print(f"\nDataset Info:")
print(f"  Train samples : {train_gen.samples}")
print(f"  Val samples   : {val_gen.samples}")
print(f"  Test samples  : {test_gen.samples}")
print(f"  Classes       : {NUM_CLASSES}")
print(f"  Steps per epoch: {len(train_gen)}\n")

# ===============================
# MODEL (SIMPLIFIED)
# ===============================
print("🏗️  Building model...\n")

model = Sequential([
    # Block 1
    Conv2D(32, (3,3), activation="relu", padding="same", input_shape=(64,64,3)),
    Conv2D(32, (3,3), activation="relu", padding="same"),
    MaxPooling2D(2,2),
    Dropout(0.25),
    
    # Block 2
    Conv2D(64, (3,3), activation="relu", padding="same"),
    Conv2D(64, (3,3), activation="relu", padding="same"),
    MaxPooling2D(2,2),
    Dropout(0.25),
    
    # Block 3
    Conv2D(128, (3,3), activation="relu", padding="same"),
    MaxPooling2D(2,2),
    Dropout(0.4),
    
    # Dense
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print(f"Total params: {model.count_params():,}\n")

# ===============================
# CALLBACKS
# ===============================
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

# ===============================
# TRAINING
# ===============================
print("="*60)
print("🚀 TRAINING START...")
print("="*60 + "\n")

start_time = time.time()

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    workers=4,           # ✅ Multi-threading
    use_multiprocessing=False,
    verbose=1
)

elapsed = time.time() - start_time
epochs_trained = len(history.history['loss'])
time_per_epoch = elapsed / epochs_trained

print("\n" + "="*60)
print("⏱️  TRAINING TIME")
print("="*60)
print(f"Total time    : {elapsed/60:.1f} minutes")
print(f"Epochs trained: {epochs_trained}")
print(f"Time per epoch: {time_per_epoch/60:.1f} minutes")
print("="*60)

if time_per_epoch < 120:
    print("\n✅ Training speed is GOOD!")
elif time_per_epoch < 300:
    print("\n⚠️  Training speed is acceptable for CPU")
else:
    print("\n❌ Training is still too slow")
    print("   Consider: smaller image size or simpler model")

# ===============================
# EVALUATE
# ===============================
print("\n" + "="*60)
print("📊 EVALUATING ON TEST SET...")
print("="*60 + "\n")

test_loss, test_acc = model.evaluate(test_gen, verbose=1)

print("\n" + "="*60)
print("📊 FINAL RESULTS")
print("="*60)
print(f"Test Loss    : {test_loss:.4f}")
print(f"Test Accuracy: {test_acc*100:.2f}%")
print("="*60)

# ===============================
# DETAILED METRICS
# ===============================
print("\n📋 Calculating detailed metrics...")

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Reset generator
test_gen.reset()

# Predictions
y_pred = model.predict(test_gen, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_gen.classes

# Classification report
LABELS = list("ABCDEFGHIKLMNOPQRSTUVWXY")[:NUM_CLASSES]
report = classification_report(y_true, y_pred_classes, target_names=LABELS, digits=4)

print("\n" + "="*60)
print("📋 CLASSIFICATION REPORT")
print("="*60)
print(report)

# Overall metrics
from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_classes, average='weighted')

print("="*60)
print("🎯 OVERALL METRICS (Weighted)")
print("="*60)
print(f"Accuracy : {test_acc*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall   : {recall*100:.2f}%")
print(f"F1-Score : {f1*100:.2f}%")
print("="*60)

# ===============================
# SAVE MODEL
# ===============================
os.makedirs("model", exist_ok=True)
model.save("model/sign_language_cnn.h5")
print("\n✅ Model saved: model/sign_language_cnn.h5")

# Save metrics to file
os.makedirs("results", exist_ok=True)
with open("results/training_summary.txt", "w") as f:
    f.write("="*60 + "\n")
    f.write("TRAINING SUMMARY\n")
    f.write("="*60 + "\n\n")
    f.write(f"Image Size: {IMG_SIZE}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Epochs: {epochs_trained}/{EPOCHS}\n")
    f.write(f"Training Time: {elapsed/60:.1f} minutes\n")
    f.write(f"Time per Epoch: {time_per_epoch/60:.1f} minutes\n\n")
    f.write(f"Test Accuracy: {test_acc*100:.2f}%\n")
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Precision: {precision*100:.2f}%\n")
    f.write(f"Recall: {recall*100:.2f}%\n")
    f.write(f"F1-Score: {f1*100:.2f}%\n\n")
    f.write("="*60 + "\n")
    f.write("CLASSIFICATION REPORT\n")
    f.write("="*60 + "\n\n")
    f.write(report)

print("✅ Summary saved: results/training_summary.txt")

print("\n" + "="*60)
print("✨ TRAINING COMPLETED!")
print("="*60 + "\n")