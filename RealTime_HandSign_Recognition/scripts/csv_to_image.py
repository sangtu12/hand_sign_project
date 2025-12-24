import os
import cv2
import pandas as pd
import numpy as np

# ===============================
# PATH
# ===============================
CSV_TRAIN = "raw/sign_mnist_train.csv"
CSV_TEST  = "raw/sign_mnist_test.csv"

OUTPUT_TRAIN = "dataset/images/train"
OUTPUT_TEST  = "dataset/images/test"

IMG_SIZE = 28

# ===============================
# FUNGSI KONVERSI
# ===============================
def csv_to_images(csv_path, output_dir):
    df = pd.read_csv(csv_path)

    # Kolom pertama = label
    labels = df.iloc[:, 0].values
    pixels = df.iloc[:, 1:].values

    print(f"[INFO] Total data: {len(labels)}")

    for idx, (label, pixel_row) in enumerate(zip(labels, pixels)):
        label_dir = os.path.join(output_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)

        # reshape ke 28x28
        image = pixel_row.reshape(IMG_SIZE, IMG_SIZE)

        # convert ke uint8 (0–255)
        image = image.astype(np.uint8)

        filename = f"{idx}.png"
        filepath = os.path.join(label_dir, filename)

        cv2.imwrite(filepath, image)

        if idx % 1000 == 0:
            print(f"[INFO] Saved {idx} images")

    print(f"[DONE] Dataset disimpan di: {output_dir}")

# ===============================
# EKSEKUSI
# ===============================
csv_to_images(CSV_TRAIN, OUTPUT_TRAIN)
csv_to_images(CSV_TEST, OUTPUT_TEST)
