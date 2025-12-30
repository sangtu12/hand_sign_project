import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm


# Dapatkan directory script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# ===============================
# PATH
# ===============================
CSV_TRAIN = "dataset/raw/sign_mnist_train.csv"
CSV_TEST  = "dataset/raw/sign_mnist_test.csv"

OUTPUT_TRAIN = "dataset/images/train"
OUTPUT_TEST  = "dataset/images/test"

IMG_SIZE = 28  #

# ===============================
# FUNGSI KONVERSI CSV KE IMAGE (DENGAN KONVERSI KE RGB)
# ===============================
def csv_to_images(csv_path, output_dir):
    """
    Konversi CSV Sign MNIST ke folder images dengan struktur:
    output_dir/
        0/
            0.png
            1.png
            ...
        1/
            ...
    
    PENTING: 
    - Dataset asli adalah grayscale (28x28)
    - Kita convert ke RGB (28x28x3) agar konsisten dengan training
    - Training akan resize ke 128x128 secara otomatis
    """
    print(f"\n{'='*60}")
    print(f"📂 Processing: {csv_path}")
    print(f"📁 Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # Baca CSV
    df = pd.read_csv(csv_path)
    
    # Kolom pertama = label, sisanya = pixel values (784 kolom untuk 28x28)
    labels = df.iloc[:, 0].values
    pixels = df.iloc[:, 1:].values
    
    total_images = len(labels)
    print(f"📊 Total data: {total_images:,}")
    
    # Hitung distribusi per kelas
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\n📋 Distribusi per kelas:")
    for label, count in zip(unique, counts):
        # Mapping label ke huruf (0=A, 1=B, dst, skip J=9)
        letter = chr(65 + label) if label < 9 else chr(65 + label + 1)
        print(f"   Class {label:2d} ({letter}): {count:5d} images")
    
    print(f"\n🔄 Memulai konversi...")
    
    # Proses setiap baris dengan progress bar
    for idx, (label, pixel_row) in enumerate(tqdm(zip(labels, pixels), 
                                                   total=total_images, 
                                                   desc="Converting")):
        # Buat folder untuk setiap label
        label_dir = os.path.join(output_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)
        
        # Reshape pixel values ke 28x28
        image_gray = pixel_row.reshape(IMG_SIZE, IMG_SIZE).astype(np.uint8)
        
        # PENTING: Convert grayscale ke RGB (agar channel = 3)
        # Ini konsisten dengan training yang expect input (128, 128, 3)
        image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
        
        # Simpan sebagai PNG (akan disimpan sebagai RGB image)
        filename = f"{idx}.png"
        filepath = os.path.join(label_dir, filename)
        
        cv2.imwrite(filepath, image_rgb)
    
    print(f"\n✅ Selesai! Dataset disimpan di: {output_dir}")
    print(f"{'='*60}\n")

# ===============================
# FUNGSI VERIFIKASI
# ===============================
def verify_dataset(output_dir):
    """
    Verifikasi hasil konversi dan cek sample image
    """
    print(f"\n🔍 Verifikasi dataset: {output_dir}")
    print(f"{'='*60}")
    
    total_files = 0
    
    # Get all class folders (sort as integers)
    try:
        class_folders = sorted([f for f in os.listdir(output_dir) 
                               if os.path.isdir(os.path.join(output_dir, f))],
                              key=int)  # Sort sebagai integer
    except ValueError:
        # Jika ada folder non-numeric, sort as string
        class_folders = sorted([f for f in os.listdir(output_dir) 
                               if os.path.isdir(os.path.join(output_dir, f))])
    
    # Hitung jumlah file per kelas
    for class_folder in class_folders:
        class_path = os.path.join(output_dir, class_folder)
        num_files = len([f for f in os.listdir(class_path) if f.endswith('.png')])
        total_files += num_files
        
        # Mapping label ke huruf
        try:
            label_num = int(class_folder)
            letter = chr(65 + label_num) if label_num < 9 else chr(65 + label_num + 1)
            print(f"   Class {class_folder:2s} ({letter}): {num_files:5d} images")
        except ValueError:
            print(f"   Class {class_folder}: {num_files:5d} images")
    
    print(f"{'='*60}")
    print(f"📊 Total images: {total_files:,}")
    
    # Verifikasi sample image
    if class_folders:
        sample_class = class_folders[0]
        sample_class_path = os.path.join(output_dir, sample_class)
        sample_files = [f for f in os.listdir(sample_class_path) if f.endswith('.png')]
        
        if sample_files:
            sample_img_path = os.path.join(sample_class_path, sample_files[0])
            sample_img = cv2.imread(sample_img_path)
            
            print(f"\n🖼️  Sample image check:")
            print(f"   Path  : {sample_img_path}")
            print(f"   Shape : {sample_img.shape}")
            print(f"   Type  : {sample_img.dtype}")
            
            if len(sample_img.shape) == 3 and sample_img.shape[2] == 3:
                print(f"   ✅ Format: RGB (3 channels) - CORRECT!")
            elif len(sample_img.shape) == 2:
                print(f"   ⚠️  Format: Grayscale - PERLU DIUBAH KE RGB!")
            else:
                print(f"   ❌ Format: Unknown")
    
    print()
    
    # Hitung jumlah file per kelas
    for class_folder in class_folders:
        class_path = os.path.join(output_dir, class_folder)
        num_files = len([f for f in os.listdir(class_path) if f.endswith('.png')])
        total_files += num_files
        
        # Mapping label ke huruf
        label_num = int(class_folder)
        letter = chr(65 + label_num) if label_num < 9 else chr(65 + label_num + 1)
        print(f"   Class {class_folder:2s} ({letter}): {num_files:5d} images")
    
    print(f"{'='*60}")
    print(f"📊 Total images: {total_files:,}")
    
    # Verifikasi sample image (cek shape dan channel)
    if class_folders:
        sample_class = class_folders[0]
        sample_class_path = os.path.join(output_dir, sample_class)
        sample_files = [f for f in os.listdir(sample_class_path) if f.endswith('.png')]
        
        if sample_files:
            sample_img_path = os.path.join(sample_class_path, sample_files[0])
            sample_img = cv2.imread(sample_img_path)
            
            print(f"\n🖼️  Sample image check:")
            print(f"   Path  : {sample_img_path}")
            print(f"   Shape : {sample_img.shape}")
            print(f"   Type  : {sample_img.dtype}")
            
            if len(sample_img.shape) == 3 and sample_img.shape[2] == 3:
                print(f"   ✅ Format: RGB (3 channels) - CORRECT!")
            elif len(sample_img.shape) == 2:
                print(f"   ⚠️  Format: Grayscale - PERLU DIUBAH KE RGB!")
            else:
                print(f"   ❌ Format: Unknown")
    
    print()

# ===============================
# MAIN EXECUTION
# ===============================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 SIGN MNIST: CSV TO IMAGES CONVERTER")
    print("   (dengan konversi Grayscale → RGB)")
    print("="*60)
    
    # Cek file CSV exists
    if not os.path.exists(CSV_TRAIN):
        print(f"❌ Error: File {CSV_TRAIN} tidak ditemukan!")
        print(f"   Letakkan file sign_mnist_train.csv di folder 'raw/'")
        exit(1)
    
    if not os.path.exists(CSV_TEST):
        print(f"❌ Error: File {CSV_TEST} tidak ditemukan!")
        print(f"   Letakkan file sign_mnist_test.csv di folder 'raw/'")
        exit(1)
    
    # Konversi Training Set
    csv_to_images(CSV_TRAIN, OUTPUT_TRAIN)
    verify_dataset(OUTPUT_TRAIN)
    
    # Konversi Test Set
    csv_to_images(CSV_TEST, OUTPUT_TEST)
    verify_dataset(OUTPUT_TEST)
    
    print("✨ KONVERSI SELESAI! ✨")
    print("="*60)
    print("\n📝 Catatan Penting:")
    print("   ✅ Images dikonversi dari 28x28 grayscale → 28x28 RGB (3 channels)")
    print("   ✅ ImageDataGenerator akan otomatis resize ke 128x128 saat training")
    print("   ✅ Struktur: dataset/images/train/{class_id}/{image_id}.png")
    print("   ✅ Struktur: dataset/images/test/{class_id}/{image_id}.png")
    print("   ✅ Format konsisten dengan model CNN (RGB input)")
    print("\n💡 Label mapping:")
    print("   0-8  = A-I")
    print("   9    = K (J di-skip karena gesture dinamis)")
    print("   10-24 = L-Y (Z di-skip karena gesture dinamis)")
    print()