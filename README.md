# 🤟 Real-Time Hand Sign Recognition (MediaPipe + TensorFlow)

Proyek ini merupakan sistem **pengenalan bahasa isyarat tangan secara real-time** menggunakan **MediaPipe Hands** untuk ekstraksi landmark dan **TensorFlow/Keras** untuk klasifikasi huruf.

Project ini dirancang **lintas platform** dan dapat dijalankan di **Windows maupun macOS**.

---

## 📌 Fitur Utama

- Deteksi tangan real-time (MediaPipe / YOLO)
- Klasifikasi bahasa isyarat alfabet (A–Z)
- **Dua pendekatan model**:

  - CNN berbasis gambar (image-based)
  - CNN berbasis landmark (landmark-based)

- Eksperimen YOLO untuk object detection
- Sistem modular & mudah dikembangkan

---

## 🗂 Struktur Folder

Struktur folder lengkap sesuai seluruh proses project (YOLO, CNN Image, dan Landmark):

```
RealTime_HandSign_Recognition/
│
├── dataset/
│   ├── raw/                     # Dataset mentah (CSV Kaggle, dll)
│   ├── images/                  # Dataset image (eksperimen CNN / YOLO)
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── landmarks/               # Dataset landmark MediaPipe (CSV per huruf)
│
├── model/
│   ├── sign_language_cnn.h5     # Model CNN image-based
│   ├── hand_landmark_model.h5   # Model CNN landmark-based
│   └── yolo/                    # (Opsional) Model YOLO
│
├── scripts/
│   ├── collect_data.py          # Capture image dari webcam
│   ├── auto_label_yolo.py       # Auto labeling bounding box YOLO
│   ├── csv_to_image.py          # Konversi CSV Kaggle → image
│   ├── train_cnn.py             # Training CNN berbasis image
│   ├── realtime_sign.py         # Realtime sign (CNN image-based)
│   ├── collect_landmark.py      # Rekam data landmark MediaPipe
│   ├── train_landmark_model.py  # Training CNN landmark
│   └── realtime_landmark.py     # Realtime landmark recognition
│
├── data.yaml                    # Konfigurasi dataset YOLO
├── requirements.txt
└── README.md
```

---

## 🛠 Teknologi yang Digunakan

- Python 3.9 – 3.11
- TensorFlow / Keras 2.13
- MediaPipe 0.10.21
- OpenCV 4.x
- NumPy 1.24.3
- Pandas
- Scikit-learn
- (Opsional) YOLO / Ultralytics

---

## 📦 Instalasi Environment

### 1️⃣ Buat Virtual Environment

```bash
python -m venv .venv
```

Aktivasi:

- **Windows**

```bash
.venv\Scripts\activate
```

- **macOS / Linux**

```bash
source .venv/bin/activate
```

---

### 2️⃣ Install Dependencies

#### 🔹 Windows / Mac Intel

```bash
pip install tensorflow==2.13.0
```

#### 🔹 Mac Apple Silicon (M1/M2/M3)

```bash
pip install tensorflow-macos==2.13.0
```

#### 📦 Library lainnya

```bash
pip install mediapipe==0.10.21
pip install numpy==1.24.3
pip install protobuf==4.25.3
pip install opencv-python
pip install pandas scikit-learn matplotlib
```

---

## 🎥 Langkah Penggunaan

Project ini dikembangkan melalui **dua pendekatan utama**, yaitu **YOLO (image-based)** dan **MediaPipe Landmark (final)**. Berikut penjelasan lengkap penggunaan dataset dan kode pada masing-masing pendekatan.

---

## 🔶 Pendekatan 1: Image-Based (YOLO & CNN)

Pendekatan ini merupakan **tahap awal eksplorasi**, menggunakan dataset gambar tangan dan YOLO untuk deteksi objek.

### 📁 Dataset YOLO

Struktur dataset YOLO:

```
dataset/images/
├── train/
├── val/
└── test/

dataset/labels/
├── train/
├── val/
└── test/
```

Setiap gambar memiliki file label `.txt` berformat YOLO:

```
<class_id> <x_center> <y_center> <width> <height>
```

### 📜 Kode Terkait YOLO

- `collect_data.py`

  - Mengambil gambar tangan dari webcam
  - Menyimpan ke folder dataset image

- `auto_label_yolo.py`

  - Membuat bounding box otomatis
  - Menghasilkan file label YOLO

- `data.yaml`

  - Konfigurasi dataset YOLO
  - Digunakan saat training YOLO

### 🧪 Tujuan Penggunaan YOLO

- Eksperimen object detection tangan
- Auto-label dataset
- Memahami pipeline deteksi berbasis gambar

⚠️ **Catatan**: Pendekatan ini menghasilkan deteksi tangan, namun **kurang stabil untuk klasifikasi huruf realtime**, sehingga tidak dipakai sebagai solusi akhir.

---

## 🔷 Pendekatan 2: Landmark-Based (MediaPipe + CNN)

Pendekatan ini merupakan **solusi final** karena lebih stabil dan ringan.

### 📁 Dataset Landmark

```
dataset/landmarks/
├── A.csv
├── B.csv
├── C.csv
└── D.csv
```

Setiap file CSV berisi:

- 21 titik landmark tangan
- Koordinat (x, y)
- Label huruf

### 📜 Kode Terkait Landmark

- `collect_landmark.py`

  - Merekam landmark tangan menggunakan MediaPipe
  - Menyimpan data ke CSV sesuai label

- `train_landmark_model.py`

  - Melatih CNN berbasis landmark
  - Output: `hand_landmark_model.h5`

- `realtime_landmark.py`

  - Deteksi tangan realtime
  - Ekstraksi landmark
  - Prediksi huruf

---

### 🔹 1. Rekam Data Landmark

Rekam landmark tangan untuk setiap huruf.

```bash
python scripts/collect_landmark.py
```

- Tekan tombol sesuai label (A, B, C, ...)
- Setiap gesture **HARUS konsisten**
- Data akan disimpan dalam format `.csv`

---

### 🔹 2. Training Model Landmark

```bash
python scripts/train_landmark_model.py
```

Output:

```
model/hand_landmark_model.h5
```

---

### 🔹 3. Jalankan Realtime Detection

```bash
python scripts/realtime_landmark.py
```

- Kamera akan aktif
- Tampilkan gesture di depan kamera
- Huruf akan muncul secara real-time

---

## ⚠️ Catatan Penting

- **Model landmark TIDAK menggunakan gambar mentah**
- Dataset gambar (Kaggle, MNIST, dsb) **tidak cocok langsung** untuk MediaPipe realtime
- Jika huruf selalu salah:

  - Pastikan label CSV benar
  - Data tiap huruf seimbang
  - Gesture konsisten

---

## ❗ Troubleshooting Umum

### Kamera tidak muncul

- Coba index kamera:

```python
cv2.VideoCapture(1)
```

- Pastikan izin kamera aktif (macOS)

### Hanya huruf 'A' yang muncul

- Semua data tersimpan dengan label sama
- Dataset belum di-reset

### Error model tidak ditemukan

```
OSError: No file or directory found at model/hand_landmark_model.h5
```

➡ Jalankan training terlebih dahulu

---

## 🧠 Kenapa Pakai Landmark?

Walaupun project ini **sempat menggunakan dataset gambar dan YOLO**, pendekatan landmark dipilih sebagai solusi akhir karena lebih stabil.

| Image-based CNN / YOLO  | Landmark-based MediaPipe    |
| ----------------------- | --------------------------- |
| Sensitif cahaya         | Stabil terhadap cahaya      |
| Bergantung bounding box | Berdasarkan struktur tangan |
| Dataset besar           | Dataset kecil sudah cukup   |
| Kurang stabil realtime  | Sangat cocok realtime       |

---

## 📌 Kesimpulan

✅ Sistem ini **lebih akurat dan konsisten** untuk realtime sign recognition

✅ Cocok untuk tugas kuliah, demo AI, dan penelitian dasar

---

## 👨‍💻 Catatan

Project ini dikembangkan sebagai **project pembelajaran Computer Vision & AI** menggunakan MediaPipe dan TensorFlow.

Jika ingin dikembangkan lebih lanjut:

- Tambah smoothing prediksi
- Tambah kalimat (sequence model)
- Tambah huruf J & Z (gesture dinamis)
