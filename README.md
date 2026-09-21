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
├── dataset_fer2013/            # dataset (tersimpan di luar repo: hasil_ekspresi_yolov8m/dataset_fer2013)
│   ├── train/  (47 ribu)       # CLAHE + resize 224 + penyeimbang (jijik 7000, marah/netral/senang/terkejut 6000, sedih 7500, takut 8500)
│   └── val/    (7178)          # data uji asli FER2013 (tidak seimbang, jijik 111)
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
│   └── classify/               # arsip uji validasi
├── docs/
│   ├── Dokumentasi Klasifikasi Ekspresi Wajah.docx  # 📄 dokumentasi resmi: use case, dataset, langkah pakai (screenshot asli), source code
│   ├── proposal.md             # draf proposal format PIDI Digdaya 2026
│   ├── video_script.md         # naskah video demo + suara
│   ├── README_ID.md            # README versi awal (Bahasa Indonesia)
│   ├── confusion_matrix.png / confusion_matrix_v3.png
│   ├── results.png / results_v3.png
│   ├── results.csv
│   └── args.yaml
├── image-test/                 # 7 foto uji manual (satu per ekspresi)
├── export_v3/                  # artefak export model V3
├── LICENSE                     # MIT © 2026 Barksuy
├── README.md
└── pyproject.toml
```

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

> Hasil V2: `yolov8m-ekpresi/results.csv:51` Top1 terbaik 71,22% (val_loss 1,10), val_loss terbaik 0,883 epoch14. Overfit setelah epoch14 (train 0,17 vs val 1,31).

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
