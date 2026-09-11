# Draft Proposal — Facial Expression Monitoring (Format PIDI Digdaya Hackathon 2026)
> Ikuti `template_pdf (1).pdf` & `prompt gpt.docx` (VisionGuard MBG) — isian di bawah sudah disesuaikan ke **Klasifikasi Ekspresi Wajah YOLOv8m (FER2013)**. Tinggal copy-paste ke PDF template, font Arial/Calibri 11pt, spasi 1-1.15, nama file `Tim_NamaKetua_Proposal Hackathon dan Digdaya 2026.pdf`.

---

### 1. Team Identity
- **Nama Tim:** VisionGuard / Herfandi Team
- **Ketua Tim:** Herfandi
- **Kontak:** [isi WA]
- **Email:** [isi email]
- **Institusi:** [isi kampus]
- **Link Portofolio:** `https://github.com/herfandi/facial-expression-classification` (atau `hasil_ekspresi_yolov8m`)
- **Anggota & Peran:**
  - Herfandi — Team Lead / Product Strategist
  - Anggota 2 — AI & Computer Vision Engineer (YOLOv8m, preprocessing CLAHE)
  - Anggota 3 — Backend & Dashboard Developer (Streamlit, tracking)
  - Anggota 4 — UI/UX & Business Analyst
- **Ringkasan Tim (≤600 karakter):**
  Tim multidisiplin AI/computer vision + web. Relevan karena mampu menerjemahkan kebutuhan monitoring ekspresi (kelas/klinik) menjadi platform real-time: deteksi wajah → klasifikasi 7 emosi → dashboard prioritas. Sudah buktikan pipeline FER2013: preprocessing 224px + CLAHE, balance 6000/kelas, YOLOv8m 71.2% Top1.

### 2. Executive Summary Project (≤600 karakter)
Platform monitoring ekspresi siswa berbasis computer vision & AI. Mengolah foto/video/CCTV → deteksi wajah (MediaPipe) → klasifikasi YOLOv8m-cls 7 emosi → risk score & dashboard persentase. Mendukung tema **Percepatan Layanan Publik & Ekonomi Kreatif**: membantu guru/konselor deteksi emosi negatif secara objektif, cepat, berbasis bukti visual, menggantikan observasi manual. Dampak: layanan pendidikan lebih responsif, data untuk intervensi dini.

### 3. Problem Statement
- **Problem Statement Utama:** Percepatan Layanan Publik, Ekonomi Kreatif, dan Ekspor Jasa Digital
- **Sub-Problem:** Digitalisasi Layanan Publik & Pariwisata — Platform Monitoring & Analitik (adaptasi dari “Perizinan dan Pengawasan Vendor MBG”)
- **Tujuan Utama (≤800):** Membangun sistem real-time yang akurat (Top1 >70% FER, Top5 99%) untuk 7 ekspresi, dengan interface sederhana (upload/webcam/video), tracking kumulatif, dan validasi 7-pose agar sekolah bisa monitor wellbeing siswa tanpa alat mahal.

### 4. Problem Definition
- **Apa Masalah Utamanya? (≤1000):** Pengamatan emosi siswa masih manual, subjektif, tidak terekam. FER2013 benchmark hanya ~65-73% jika tanpa preprocessing matang; dataset imbalance (jijik 547 vs happy 8989 asli), grayscale flat, annotasi noisy → model gampang bias ke `senang`. Tanpa CLAHE+balance+224px, akurasi stagnan 67%.
- **Siapa Terdampak? (≤700):** Guru BK, wali kelas, sekolah, klinik remaja. Skala: 1 guru awasi 30+ siswa, butuh prioritas siapa yang sedih/takut dominan.
- **Bukti Masalah (≤800):** FER2013 publik, literatur akurasi 65-76%. Eksperimen kami: V0 YOLOv8n 96px 67.14% (`model_ekspresi/results.csv:50`), V1 YOLOv8s 96px 68.63% (`hasil_ekspresi_yolov8s/.../results.csv:67`), V2 YOLOv8m 224px 71.22% (`hasil_ekspresi_yolov8m/.../results.csv:51`). Confusion: takut 51%, sedih 59% — butuh validasi pose manual.

### 5. Proposed Solution — Solusi Inti (≤900)
Sistem 3 lapis: (1) Preprocessing — grayscale→RGB+CLAHE+resize 224 + augment heavy+balance 6000/kelas; (2) Model — YOLOv8m-cls fine-tune 150ep AdamW cos_lr dropout0.2; (3) Aplikasi — Streamlit 4 mode: Upload Gambar, Webcam Realtime, Test 7 Ekspresi, Upload Video (1 fps). Output: bbox+label+confidence+bar chart+ tabel persentase per wajah+ rekap dominan.

### 6. Bagaimana Solusi Bekerja? (≤900, Input→Proses→Output)
**Input:** foto JPG/PNG, frame webcam, atau video mp4. **Proses:** (a) Face detection MediaPipe (fallback HaarCascade), crop wajah; (b) YOLOv8m infer → probs top1/top5; (c) Tracking `Wajah_1..N` Counter di `st.session_state`; (d) Aggregasi 1 frame/detik untuk video. **Output:** annotated frame, label Indo+EN+emoji, confidence warna (≥70% hijau), distribusi probabilitas, tabel `%` per siswa, dan validasi batch 7 foto (✅/❌ per ekspresi, akurasi manual).

### 7. Impact & Outcome — Manfaat Utama (≤800)
Guru dapat prioritas inspeksi: siswa dominan sedih/takut → follow-up. Objektif (berbasis confidence), terekam (log), cepat (real-time). Mengurangi bias observasi, memberi data longitudinal wellbeing. Sekolah bisa ekspor rekap untuk laporan.

### 8. Dampak Jangka Pendek & Menengah (≤600)
**Pendek (1-3 bulan):** pilot 1-2 kelas, kumpulkan 7-pose per siswa untuk re-kalibrasi threshold. **Menengah (6-12 bulan):** integrasi CCTV kelas, notifikasi jika sedih>40% dalam sesi, dashboard sekolah, dataset lokal incremental untuk push 71%→78%.

### 9. Innovation & Differentiation — Keunikan (≤700)
- End-to-end FER: CLAHE+balance 6k+224px+m-cls (naik 4% vs 96px) — tidak sekadar pakai model jadi.
- Validasi 7-pose built-in (uji 7 foto ekspresi, langsung hitung akurasi manual) — fitur yang tidak ada di corn-leaf example.
- Monitoring kumulatif multi-wajah + video 1 fps — mirip corn-leaf tapi untuk emosi.
- Bilingual label Indo (jijik/marah/...) agar sesuai kurikulum lokal.

### 10. Posisi vs Produk Ada (≤700)
Produk komersial (Affectiva, Azure Face) mahal, cloud, EN-only. Pendekatan open-source (FER CNN biasa) akurasi 65-68% dan tanpa tracking. Solusi ini melengkapi: on-device YOLOv8m, Indo-label, open, bisa self-host di sekolah, akurasi FER kompetitif (71.2%) dengan code & notebook lengkap untuk reproduksi.

### 11. Technical Approach — Teknologi Utama (≤700)
YOLOv8m-cls (Ultralytics), OpenCV CLAHE, Streamlit, MediaPipe FaceDetection, Python. Alasan: YOLO cepat, akurat, mudah export ONNX; Streamlit tercepat untuk demo interaktif seperti corn-leaf (`streamlit-app/main.py:28`).

### 12. Pemilihan & Penggunaan Teknologi (≤600)
Pilih YOLOv8m karena +2-4% vs s/n (eksperimen). AdamW+cos_lr lebih stabil 150ep. 224px penting untuk detail wajah FER 48px. MediaPipe untuk deteksi wajah real-time ringan; fallback Haar jika mediapipe tidak ada.

### 13. Algoritma Solusi (≤700)
Fine-tune YOLOv8m-cls pretrained (transfer learning). Loss: CrossEntropy + label_smoothing 0.1. Augment: RandAugment+erasing 0.4+degrees15/shear2. Inference: argmax probs. Evaluasi: Top1/Top5, confusion matrix, classification_report. Tidak perlu blockchain untuk MVP; bisa tambah log hash jika perlu audit.

### 14. Data/Input Utama (≤900)
FER2013 Kaggle (msambare/fer2013): train/test split. Preprocess: `klasifikasi-ekspresi-v2.ipynb:4` konversi grayscale→BGR+CLAHE+resize cubic. Balance: `cell 6` 6000/kelas via augment heavy (flip/rot ±15°/brightness/blur/cutout) + undersample happy. Val 7178 imbalance (jijik 111) dipakai apa adanya untuk evaluasi jujur. Kualitas: noisy label, tapi CLAHE & balance mengurangi bias. Sumber jelas, reproducible.

### 15. Keamanan & Skalabilitas (≤600)
Data wajah diproses lokal (tidak upload cloud), bisa blur jika perlu privacy. Streamlit session_state tidak simpan permanen, tombol Reset. Skalabilitas: tambah batch inference, export ONNX (`model.export(format="onnx")`), deploy Docker, ganti ke YOLOv8s untuk edge device.

### 16. Implementation Feasibility — Status Inovasi
**Prototype** — sudah ada: notebook 15 cells, `hasil_ekspresi_yolov8m/yolov8m-ekpresi/weights/best.pt`, `streamlit_app.py` 4 mode, confusion 71.15%. Siap demo.

### 17. Apakah Realistis? (≤700)
Ya: tim sudah buktikan 3 iterasi (n→s→m) naik 67→71%. Stack open-source, run di laptop CPU/GPU. Dataset tersedia, training 150ep di Kaggle T4 (~2 jam). Sekolah hanya butuh webcam + laptop.

### 18. Tahapan Pengembangan (≤900)
**Tahap 1 (Minggu 1-2):** finalisasi dataset 6k/kelas + training V2 (done). **Tahap 2 (Minggu 3):** polish Streamlit (Test 7-pose, video) — done. **Tahap 3 (Minggu 4):** pilot 20 siswa, kumpulkan feedback takut/sedih. **Tahap 4 (Bulan 2):** fine-tune dengan data pilot, export ONNX, Docker. **Tahap 5 (Bulan 3):** dashboard multi-kelas & notifikasi, pilot sekolah.

### 19. Bisnis Model & Keberlanjutan (≤1000) — BMC ringkas
**Customer:** sekolah, bimbel, klinik remaja. **Value:** monitoring emosi objektif, laporan BK. **Channel:** Streamlit on-premise + mobile wrapper. **Revenue:** freemium (1 kelas gratis), subscription per sekolah / per CCTV, jasa fine-tune. **Cost:** GPU training awal, maintenance. **Partner:** dinas pendidikan, penyedia CCTV. **Keberlanjutan:** data flywheel (semakin banyak 7-pose lokal, akurasi naik), open-core (code free, hosting/support berbayar).

### 20. Attachment & Reference
- Screenshot: `yolov8m-ekpresi/results.png`, `confusion_matrix.png`, `val_batch0_pred.jpg`, Streamlit Test 7 Ekspresi (✅/❌).
- Link demo: `[isi link Streamlit Cloud / YouTube demo]`
- Repo: `https://github.com/Ikhwanand/corn-leaf-disease-classification` (referensi struktur) + repo baru `facial-expression-classification`
- Dataset: `https://www.kaggle.com/datasets/msambare/fer2013`
- Ultralytics YOLOv8 docs
