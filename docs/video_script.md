# Naskah Video Demo — Facial Expression Classification (3.5–4 menit, ada suara)
> Rekam layar + webcam + voice-over Indonesia. Ikuti alur corn-leaf Streamlit: dari `streamlit run` sampai akhir + jelaskan fungsi. Resolusi 1080p, mic jelas.

### 0. Intro (0:00-0:20) — Tampilkan Judul
**Layar:** judul besar “😊 Facial Expression Classification — YOLOv8m FER2013 (7 Ekspresi) | Herfandi Team” + `best.pt` info (Top1 71.22% Top5 99.2%).
**VO:** “Assalamualaikum, saya Herfandi. Kali ini saya akan mendemonstrasikan aplikasi klasifikasi 7 ekspresi wajah berbasis YOLOv8m dan Streamlit, hasil fine-tune FER2013 dengan akurasi 71 persen.”

### 1. Menjalankan Aplikasi (0:20-0:45)
**Aksi:** buka terminal VSCode / PowerShell, `cd facial-expression-classification/streamlit-app`, `streamlit run main.py`. Tampilkan browser `localhost:8501` terbuka.
**VO:** “Cara menjalankannya cukup `streamlit run main.py`. Aplikasi akan terbuka di localhost 8501. Di sidebar ada info model: YOLOv8m input 224, 7 kelas Indonesia, dan panduan penggunaan.”

### 2. Mode 1 — Upload Gambar Bebas (0:45-1:30)
**Aksi:** pilih `📷 Upload Gambar (bebas)`, upload 1 foto wajah senang (close-up terang), tunggu `Menganalisis...`, tunjukkan bbox hijau `senang 98%`, confidence progress, bar chart 7 probabilitas (senang tertinggi). Upload foto kedua multi-wajah (2 orang netral+sedih) tunjukkan 2 bbox.
**VO:** “Mode pertama upload gambar bebas. Kita coba foto senang — model memprediksi senang 98 persen dengan bar chart. Untuk multi-wajah, sistem deteksi dua wajah dan klasifikasi masing-masing. Jika takut atau sedih, akan ada catatan bahwa kelas ini memang sering tertukar.”

### 3. Mode 2 — Test 7 Ekspresi (1:30-2:30) — INTI VALIDASI
**Aksi:** pindah ke `🧪 Test 7 Ekspresi`. Tampilkan 7 kolom `jijik/marah/netral/sedih/senang/takut/terkejut`. Upload 7 foto pose sendiri (satu per kolom, preview muncul). Klik `▶️ Jalankan Validasi 7 Ekspresi`. Tunjukkan tabel `Expected vs Prediksi vs Conf vs Hasil ✅/❌`, metric `Akurasi Manual 85.7% (6/7)`, detail per foto + bar chart untuk yang salah, jelaskan “takut terprediksi sedih karena confusion 17%”.
**VO:** “Mode paling penting untuk membuktikan apakah 7 ekspresi benar-benar sesuai. Saya upload tujuh pose — jijik sampai terkejut — lalu jalankan validasi. Sistem membandingkan prediksi dengan ekspektasi dan memberi centang silang. Disini kita lihat enam dari tujuh benar, yang salah hanya takut yang terprediksi sedih — memang sesuai confusion matrix model V2.”

### 4. Mode 3 — Webcam Realtime (2:30-3:10)
**Aksi:** pilih `🎥 Webcam Realtime`, centang `Aktifkan Kamera`, tunjukkan live bbox berubah saat kamu senyum → netral → terkejut. Di kanan tunjukkan tabel `Wajah_1` persentase kumulatif (senang 60%, terkejut 20%…) dan bar chart bergerak.
**VO:** “Mode webcam untuk monitoring real-time di kelas. Setiap frame dideteksi wajahnya, diprediksi, dan diakumulasi di tabel sebelah. Cocok untuk melihat distribusi emosi selama sesi.”

### 5. Mode 4 — Upload Video (3:10-3:40)
**Aksi:** pilih `🎞️ Upload Video`, upload `test.mp4` 10 detik, set `Sample 1 fps`, klik `▶️ Analisis`, tunjukkan progress, expand per detik `Detik 0 senang 92%`, akhir tunjukkan `Dominan: senang (5/10 detik)` + tabel detik.
**VO:** “Terakhir upload video — sistem ambil satu frame per detik seperti pada proyek corn-leaf, lalu rekap ekspresi dominan. Hasilnya bisa dipakai untuk laporan.”

### 6. Penutup + Code & Dokumen (3:40-4:00)
**Aksi:** tampilkan VSCode struktur repo (`models/best.pt`, `nb-1.ipynb`, `streamlit-app/main.py`, `README.md`), scroll cepat `nb-1.ipynb` 15 cells, tunjukkan `results.png` & `confusion_matrix.png`.
**VO:** “Semua code, notebook training 15 cell, model, dan dokumentasi ada di GitHub dengan struktur mirip corn-leaf disease. Dokumen proposal mengikuti template PIDI Digdaya. Sekian demo — terima kasih, wassalamualaikum.”

---
**Checklist take video:**
- [ ] Layar 1920×1080, font jelas, cursor terlihat
- [ ] Mic on, VO sesuai naskah, jangan terlalu cepat
- [ ] Siapkan 7 foto pose + 1 video 10s sebelum take
- [ ] Export mp4, upload ke YouTube unlisted, link taruh di proposal & README
- [ ] Tampilkan di akhir: `streamlit run main.py` + `pip install -r requirements.txt`
