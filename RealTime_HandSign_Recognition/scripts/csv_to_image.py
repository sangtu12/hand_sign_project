import os, cv2
import pandas as pd
import numpy as np

def csv_to_images(csv_path, out_dir):
    df = pd.read_csv(csv_path)

    labels = df['label']
    pixels = df.drop('label', axis=1).values

    for i, (label, pixel) in enumerate(zip(labels, pixels)):
        class_name = chr(label + 65)  # 0=A, 1=B, ...
        class_dir = os.path.join(out_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        img = pixel.reshape(28, 28).astype(np.uint8)
        img = cv2.resize(img, (64, 64))
        cv2.imwrite(f"{class_dir}/{i}.png", img)
