# Alur Video Demo — Klasifikasi Ekspresi Wajah (ringkasan)

> Versi lengkap dengan narasi siap baca + tangkapan layar acuan ada di
> **`docs/Naskah Video Demo Klasifikasi Ekspresi Wajah.docx`**.
> Ini hanya ringkasan alurnya untuk dibaca cepat.

Target: ±4,5 menit · voice-over Bahasa Indonesia · rekam layar 1080p + mic jelas.

| # | Menit | Adegan | Inti |
|---|---|---|---|
| 0 | 0:00–0:25 | Intro & konteks | Judul, 7 ekspresi, akurasi **69,59% top-1 / 98,98% top-5** |
| 1 | 0:25–0:55 | Menjalankan aplikasi | `streamlit run main.py` → halaman utama + sidebar pilihan model |
| 2 | 0:55–1:40 | Tab 📤 Analisis Gambar | Unggah foto senang → bbox hijau `senang 99,1%` + bar chart 7 kelas |
| 3 | 1:40–2:50 | Tab 🧪 Uji 7 Ekspresi | 7 foto → validasi **85,7% (6/7 benar)**; salah: marah → takut |
| 4 | 2:50–3:35 | Tab 🎥 Kamera Langsung | Mulai kamera → bbox + label live berubah + rekap sesi |
| 5 | 3:35–4:05 | Tab 🎞️ Analisis Video | Video 6 detik → frame/detik → ekspresi dominan + garis waktu |
| 6 | 4:05–4:45 | Tab 📊 Info Model + Penutup | Properti model, recall per kelas, confusion matrix, kurva → repositori GitHub |

## Aturan angka (wajib konsisten)
- Model final: **Top-1 69,59% · Top-5 98,98%** (uji 7.178 gambar asli FER2013, seed 42) — **jangan** ucapkan 71% (itu eksperimen lama).
- Macro avg recall **68,31%**. Baseline V3: 62,93% (150 epoch) → fine-tune: 69,59% (60 epoch).
- Recall per kelas: senang 86,5 · terkejut 82,6 · netral 69,1 · jijik 68,5 · marah 61,6 · sedih 58,7 · takut 51,3.
- Uji manual 7 foto = **85,7% (6/7)**, marah terbaca takut — hasil deterministik bila memakai foto `image-test/` yang sama.

## Checklist take
- [ ] File: 7 foto `image-test/` + `contoh_upload.jpg` + `uji_video.mp4`
- [ ] `cd streamlit-app` → `streamlit run main.py` → buka http://localhost:8501
- [ ] Layar 1080p, cursor terlihat, mic jelas
- [ ] Ubah ekspresi 3× saat adegan kamera (netral → senyum → terkejut)
- [ ] Export mp4 → (opsional) YouTube unlisted → taruh link di proposal & README