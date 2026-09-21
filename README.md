# 😊 Klasifikasi Ekspresi Wajah — YOLOv8m (FER2013)

> Aplikasi untuk mengenali 7 ekspresi wajah berbasis **YOLOv8m-cls** + Streamlit. Deteksi wajah pakai **OpenCV YuNet DNN** (cadangan MediaPipe & Haar Cascade, file model dibundel di project), lalu klasifikasi ekspresi dengan persentase keyakinan & rekap kumulatif — strukturnya mirip `corn-leaf-disease-classification` (Ikhwanand) tapi disesuaikan untuk pemantauan ekspresi siswa.

## 🎯 Fitur
- Deteksi wajah real-time dengan **bounding box live** (OpenCV YuNet DNN; cadangan MediaPipe & Haar Cascade)
- Klasifikasi 7 ekspresi: `jijik`, `marah`, `netral`, `sedih`, `senang`, `takut`, `terkejut` — ukuran masukan 224×224
- 5 tab modern: **Analisis Gambar**, **Kamera Langsung**, **Uji 7 Ekspresi (validasi)**, **Analisis Video**, **Info Model** (confusion matrix & kurva training)
- Pemilihan checkpoint model langsung dari sidebar (model final / V3 baseline / fine-tune)
- Kamera real-time: bounding box + label + persentase tampil langsung, status FPS, kartu ekspresi live
- Rekap kamera otomatis (jumlah frame, ekspresi dominan, keyakinan rata-rata)
- Akurasi Top-1 **69,59%** / Top-5 **98,98%** (validasi otomatis 7.178 gambar uji asli FER2013, seed 42 deterministik), lengkap dengan matriks kebingungan & laporan klasifikasi
- Siap pakai CPU/GPU otomatis (Ultralytics YOLO)

## 🧠 Ekspresi yang Didukung — Model Final (V3+Fine-Tune)

Recall per kelas dari **validasi 7.178 gambar uji asli FER2013** (seed 42, hasil persis terekam di notebook):

| Label | Emoji | Precision | Recall | F1 | Data uji |
|---|---|---|---|---|---|
| senang | 😄 | **89,1%** | **86,5%** | **87,8%** | 1.774 |
| terkejut | 😯 | 79,3% | 82,6% | 80,9% | 831 |
| jijik | 🤢 | **85,4%** | 68,5% | 76,0% | 111 |
| netral | 😐 | 61,3% | 69,1% | 65,0% | 1.233 |
| marah | 😠 | 61,3% | 61,6% | 61,5% | 958 |
| sedih | 😢 | 57,8% | 58,7% | 58,3% | 1.247 |
| takut | 😨 | 59,3% | 51,3% | 55,0% | 1.024 |

> **Akurasi 69,59%** · macro avg recall **68,31%** · weighted avg recall **69,59%** · **Top-5 98,98%**

> Pola salah paling sering (dari confusion matrix final): `takut→sedih 17,0%`, `sedih→netral 16,0%`, `netral→sedih 15,2%`, `marah→sedih 12,4%` — wajar untuk FER2013 yang memang sulit dibedakan manusia pun.

## 🏗️ Struktur Proyek
```
klasifikasi-ekspresi-wajah/
├── models/
│   ├── best.pt                 # YOLOv8m-cls model final (V3+Fine-Tune) — 69,59% Top1 / 98,98% Top5
│   └── yolov8m-cls.pt          # model dasar pra-latih
├── streamlit-app/
│   ├── main.py                 # 🆕 UI modern — 5 tab + pemilih model + info model
│   ├── requirements.txt
│   ├── face_detection_yunet_2023mar.onnx   # deteksi wajah utama (YuNet DNN)
│   ├── haarcascade_frontalface_default.xml # cadangan Haar Cascade
│   └── blaze_face_short_range.tflite       # cadangan MediaPipe (opsional)
├── nb-klasifikasi-ekspresi-wajah.ipynb  # 🆕 SATU notebook lengkap: platform→dataset→training V3→fine-tune→validasi→export→demo, dengan LOG EKSEKUSI ASLI di tiap sel (bukti)
├── runs/
│   ├── yolov8m-v3/             # artefak training V3 baseline (best 62,93%)
│   ├── yolov8m-v3-ft/          # artefak fine-tune final (best 69,59%)
│   └── classify/               # arsip uji validasi (val, val-2…val-9, dll.)
├── docs/
│   ├── Dokumentasi Klasifikasi Ekspresi Wajah.docx  # 📄 dokumentasi resmi: use case, dataset, langkah pakai (screenshot asli), source code
│   ├── Naskah Video Demo Klasifikasi Ekspresi Wajah.docx  # 🎬 alur video per menit + narasi siap baca + gambar acuan
│   ├── proposal.md             # draf proposal format PIDI Digdaya 2026
│   ├── video_script.md         # ringkasan alur video demo (versi teks)
│   ├── README_ID.md            # README versi awal (Bahasa Indonesia)
│   ├── Jonathan Corn Disease Classifier.docx  # referensi struktur dokumentasi teman
│   ├── confusion_matrix.png / confusion_matrix_v3.png
│   ├── results.png / results_v3.png
│   ├── results.csv
│   └── args.yaml
├── image-test/                 # 7 foto uji manual (satu per ekspresi)
├── export_v3/                  # hasil export model V3 (confusion matrix)
├── LICENSE                     # MIT © 2026 Barksuy
├── README.md
└── pyproject.toml
```
> 📁 Dataset FER2013 **tidak disimpan di repo ini** (terlalu besar) — berada di `hasil_ekspresi_yolov8m/dataset_fer2013` (train 47 ribu, val 7.178). Setiap run menyimpan konfigurasi lengkap di `runs/*/args.yaml`.

## 🚀 Cara Memulai

### Kebutuhan
- Python 3.10 atau lebih baru
- `pip`

### Instalasi
```bash
git clone https://github.com/BarskuyAurelian/klasifikasi-ekspresi-wajah.git
cd klasifikasi-ekspresi-wajah
pip install -r streamlit-app/requirements.txt
# atau
pip install -e .
```

### Menjalankan Aplikasi
```bash
cd streamlit-app
streamlit run main.py
# buka http://localhost:8501
```

## 🔧 Pelatihan — ringkasan alur (dijalankan & terekam di `nb-klasifikasi-ekspresi-wajah.ipynb`)
1. Pra-proses dataset FER2013: `grayscale→RGB + CLAHE (clip 2.0) + ubah ukuran 224×224` + penyeimbang `TARGET=6000/kelas` (kelas lemah: takut 8500, sedih 7500, jijik 7000)
2. Training V3 baseline: `YOLO("yolov8m-cls.pt").train(data, epochs=150, batch=32, AdamW, cos_lr, dropout 0.3, mixup/cutmix 0.2, freeze=9, seed=42, deterministic, amp=False)` → best **62,93%**
3. Fine-Tune Stage 2 (freeze=4, lr0=0.0002, 60 epoch): lanjut dari best V3 → best **69,59% top-1 / 98,98% top-5**
4. Validasi final: `model.val()` + confusion matrix + classification_report + top-5 confusion
5. Export → `models/best.pt` (sha256 terverifikasi identik dengan checkpoint terbaik)

> 📋 Riwayat V2 (eksperimen lama, di luar repo): Top1 71,22% namun overfit sejak epoch 14 (train 0,17 vs val 1,31) — karena itu dibuat V3 deterministik + fine-tune di bawah.

## 🧪 Pelatihan V3 — Deterministik & Stabil (`nb-klasifikasi-ekspresi-wajah.ipynb`)

Notebook baru untuk melatih ulang dengan **hasil yang bisa direproduksi identik** (seed 42 + `deterministic=True` + `amp=False`) dan konfigurasi anti-overfit:

| Perubahan V3 | Vs V2 | Alasan |
|---|---|---|
| `deterministic=True`, `amp=False`, seed penuh (numpy/torch/CUBLAS) | `deterministic=False`, `amp=True` | Hasil sama persis antar run |
| `patience=15`, `dropout=0.3`, `weight_decay=0.001` | 20 / 0.2 / 0.0005 | Stop sebelum overfit (V2 overfit sejak epoch 14) |
| `mixup=0.2`, `cutmix=0.2`, `freeze=9` (backbone, head tetap dilatih), `lr0=0.0005` | tidak ada / 0.001 | Regularisasi campuran gambar + backbone stabil |
| Balance lemah: `takut 8500`, `sedih 7500`, `jijik 7000` | semua 6000 | Naikkan recall kelas lemah (52-68%) |
| Validasi otomatis: val + confusion matrix + classification report + top-5 confusion | manual | Langsung tahu kelas mana yang keliru |

**Cara pakai (satu file, semua sel sudah berisi log eksekusi asli):**
- **Laptop / Kaggle**: buka `nb-klasifikasi-ekspresi-wajah.ipynb` → jalankan sel sesuai urutan (sel training
  bertanda ⚠️ *JANGAN JALANKAN ULANG* karena ±2 jam — hasilnya sudah terekam di output sel).
- Untuk verifikasi cepat ke dosen: cukup jalankan sel `Validasi final`, `Export`, dan `Demo`.
- Setelah export, jalankan `streamlit run streamlit-app/main.py` — app otomatis memakai `models/best.pt`.

> 💡 **Notebook adalah bukti**: setiap sel menyimpan output, tabel epoch (`results.csv`), dan grafik asli
> (`results.png`, `confusion_matrix_v3.png`); konfigurasi tiap run terekam permanen di `runs/*/args.yaml`.

> 💡 **GPU lokal**: laptop ini punya `RTX 3050 6GB`. Supaya training jalan di GPU (bukan CPU), pastikan PyTorch versi CUDA terpasang, lalu **restart kernel notebook** — notebook otomatis mengunci `yolov8m-cls.pt` + 150 epoch begitu `torch.cuda.is_available()` bernilai `True`.

## 📊 Dataset
- Sumber: `fer2013` (Kaggle msambare/fer2013) — 48×48 grayscale → diolah jadi 224×224 RGB (CLAHE clip 2.0)
- **Latih 47 ribu** (penyeimbang kelas lemah: takut 8500, sedih 7500, jijik 7000; sisanya 6000), **Uji 7.178** (tidak seimbang, jijik 111)
- Augmentasi: `randaugment`, `mosaic 1.0`, `erasing 0.4`, `mixup/cutmix 0.2`
- Evaluasi final: **Top-1 69,59% / Top-5 98,98%** · macro avg recall 68,31% (detail per kelas di tabel atas)

## 🛠️ Teknologi
- Ultralytics YOLOv8, Streamlit, OpenCV, MediaPipe, Pillow, NumPy, Pandas, scikit-learn, seaborn, matplotlib, tqdm

## 📚 Dokumentasi & Video
- `docs/proposal.md` — isian proposal PIDI Digdaya (Identitas Tim → Implementasi, tinggal salin ke template PDF)
- `docs/video_script.md` — alur demo 3-4 menit dari `streamlit run` sampai akhir + teks suara

## 📄 Lisensi
CC BY 4.0 (dataset FER2013). Kode MIT.

## 🤝 Kontribusi
Silakan ajukan Pull Request. Untuk reproduksi hasil identik, ikuti notebook `nb-klasifikasi-ekspresi-wajah.ipynb` (seed 42 + `deterministic=True` + `amp=False`).
