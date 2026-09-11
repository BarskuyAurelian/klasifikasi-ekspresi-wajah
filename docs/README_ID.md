# 😊 Klasifikasi Ekspresi Wajah — YOLOv8m (FER2013)

> Aplikasi untuk mengenali 7 ekspresi wajah berbasis **YOLOv8m-cls** + Streamlit. Deteksi wajah pakai MediaPipe/HaarCascade, lalu klasifikasi ekspresi dengan persentase keyakinan & rekap kumulatif — strukturnya mirip `corn-leaf-disease-classification` (Ikhwanand) tapi disesuaikan untuk pemantauan ekspresi siswa.

## 🎯 Fitur
- Deteksi wajah langsung (MediaPipe, cadangan HaarCascade)
- Klasifikasi 7 ekspresi: `jijik`, `marah`, `netral`, `sedih`, `senang`, `takut`, `terkejut` — ukuran masukan 224×224
- 4 mode utama: **Unggah Gambar**, **Kamera Langsung**, **🧪 Uji 7 Ekspresi (validasi)**, **Unggah Video** (1 frame/detik)
- Rekap kumulatif per `Wajah_1..N` (penghitung + tabel + diagram batang)
- Akurasi Top-1 **71,22%** / Top-5 **99,2%** (data uji FER2013 7178 gambar), validasi matriks kebingungan & laporan klasifikasi
- Siap pakai CPU/GPU otomatis (Ultralytics YOLO)

## 🧠 Ekspresi yang Didukung
| Label | Emoji | Catatan Model V2 |
|---|---|---|
| senang | 😄 | **88,5% recall** — paling baik |
| terkejut | 😯 | 82,6% |
| netral | 😐 | 72,7% |
| jijik | 🤢 | 68,4% (data uji cuma 111) |
| marah | 😠 | 62,7% |
| sedih | 😢 | 59,5% |
| takut | 😨 | **51,9%** — sering tertukar sedih/netral |

> Pola salah paling sering: `sedih→netral 17,8%`, `takut→sedih 17,2%`, `netral→sedih 13,4%` — wajar untuk FER2013.

## 🏗️ Struktur Proyek
```
klasifikasi-ekspresi-wajah/
├── dataset_fer2013/            # dataset setelah CLAHE + ubah ukuran 224 + penyeimbang 6000/kelas
│   ├── train/  (42 ribu)       # jijik/marah/netral/sedih/senang/takut/terkejut @6000
│   └── val/    (7178)          # data uji asli FER2013 (tidak seimbang, jijik 111)
├── models/
│   ├── best.pt                 # YOLOv8m-cls hasil latih ulang (20 juta param) — 71,22% Top1
│   └── yolov8m-cls.pt          # model dasar pra-latih
├── streamlit-app/
│   ├── main.py                 # aplikasi utama (4 mode, uji 7 pose)
│   ├── app_v2.py               # salinan cadangan
│   ├── haarcascade_frontalface_default.xml
│   └── requirements.txt
├── nb-1.ipynb                  # notebook pelatihan lengkap (pra-proses→latih→uji→ekspor)
├── docs/
│   ├── proposal.md             # draf proposal format PIDI Digdaya 2026
│   ├── video_script.md         # naskah video demo + suara
│   ├── results.csv / results.png
│   ├── confusion_matrix.png
│   └── args.yaml
├── README.md
└── pyproject.toml
```

## 🚀 Cara Memulai

### Kebutuhan
- Python 3.10 atau lebih baru
- `pip`

### Instalasi
```bash
git clone https://github.com/herfandi/klasifikasi-ekspresi-wajah.git
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

## 🔧 Pelatihan — `nb-1.ipynb` ringkas (15 sel, lihat file)
1. `pip install opencv-python-headless tqdm ultralytics seaborn scikit-learn`
2. Pra-proses: `grayscale→RGB + CLAHE (clip 2.0) + ubah ukuran 224×224`
3. Penyeimbang: `TARGET=6000/kelas` (augmentasi balik horizontal/rotasi±15°/kecerahan/blur/cutout)
4. `YOLO("yolov8m-cls.pt").train(data, epochs=150, imgsz=224, batch=32, AdamW, cos_lr, dropout 0.2, erasing 0.4, degrees 15, shear 2, dll)`
5. `model.val()` → Top1/Top5
6. Plot `results.png` + `confusion_matrix` + `classification_report`

> Hasil V2: `yolov8m-ekpresi/results.csv:51` Top1 terbaik 71,22% (val_loss 1,10), val_loss terbaik 0,883 epoch14. Overfit setelah epoch14 (train 0,17 vs val 1,31).

## 📊 Dataset
- Sumber: `fer2013` (Kaggle msambare/fer2013) — 48×48 grayscale → diolah jadi 224 RGB
- Latih 42 ribu seimbang, Uji 7178 (tidak seimbang). Augmentasi: `randaugment`, `mosaic 1.0`, `erasing 0.4`
- Evaluasi: trace/total 71,15%, rata-rata recall 69,5%

## 🛠️ Teknologi
- Ultralytics YOLOv8, Streamlit, OpenCV, MediaPipe, Pillow, NumPy, Pandas, scikit-learn, seaborn, matplotlib, tqdm

## 📚 Dokumentasi & Video
- `docs/proposal.md` — isian proposal PIDI Digdaya (Identitas Tim → Implementasi, tinggal salin ke template PDF)
- `docs/video_script.md` — alur demo 3-4 menit dari `streamlit run` sampai akhir + teks suara

## 📄 Lisensi
CC BY 4.0 (dataset FER2013). Kode MIT.

## 🤝 Kontribusi
Silakan ajukan Pull Request. Untuk reproduksi 90%+ butuh pembersihan data — lihat `TRAINING_90PCT.md` di sel 7 ipynb.
