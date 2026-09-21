"""
Streamlit — Klasifikasi Ekspresi Wajah (YOLOv8m-cls · FER2013 · 7 kelas)
───────────────────────────────────────────────────────────────────────
UI modern & profesional untuk menguji model:
  📤 Analisis Gambar · 🎥 Kamera · 🧪 Uji 7 Ekspresi · 🎞️ Video · 📊 Info Model

Model final: V3 + Fine-Tune (freeze=4) — Top-1 69,59% · Top-5 98,98% (seed 42, deterministik).
Bahasa: Indonesia
"""
import os
import glob
import random
import time
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

try:  # detektor wajah opsional
    import mediapipe as mp
except ImportError:
    mp = None

# ═══════════════════════════════════════════════════════════════════
#  KONFIGURASI
# ═══════════════════════════════════════════════════════════════════
REPO_ROOT = Path(__file__).resolve().parent.parent

st.set_page_config(
    page_title="Klasifikasi Ekspresi Wajah — YOLOv8m",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Pustaka model: models/best.pt (final) + semua checkpoint hasil training ──
def _cari_model():
    kandidat = []
    utama = REPO_ROOT / "models" / "best.pt"
    if utama.exists():
        kandidat.append(utama)
    kandidat += sorted(REPO_ROOT.glob("runs/*/weights/best.pt"))
    kandidat += sorted(REPO_ROOT.glob("runs/classify/*/weights/best.pt"))
    # jalur lama (backup jika repo dipindah)
    if not kandidat:
        for p in [
            r"C:\Users\Admin\Documents\Kuliah\herfandi-ml\hasil_ekspresi_yolov8m\yolov8m-ekpresi\weights\best.pt",
            r"C:\Users\Admin\Documents\Kuliah\herfandi-ml\facial-expression-classification\models\best.pt",
        ]:
            if os.path.exists(p):
                kandidat.append(Path(p))
    return list(dict.fromkeys(kandidat))  # unik, urut tetap

DAFTAR_MODEL = _cari_model()

# Metrik top-1/top-5 yang TERVERIFIKASI dari validasi final (bukan perkiraan)
METRIK_BY_FILE = {
    "best.pt":                            (69.59, 98.98),  # models/best.pt = fine-tune final (sha identik)
    "yolov8m-v3-ft":                      (69.59, 98.98),
    "yolov8m-v3":                         (62.93, 98.57),
}

INFO_KELAS = {
    "marah":    {"emoji": "😠", "warna": "#ef4444", "recall": 61.6, "support": 958,
                 "deskripsi": "Alis mengerut, rahang mengencang, bibir menekan"},
    "jijik":    {"emoji": "🤢", "warna": "#a855f7", "recall": 68.5, "support": 111,
                 "deskripsi": "Hidung mengerut, bibir atas terangkat, sering mirip marah"},
    "takut":    {"emoji": "😨", "warna": "#f59e0b", "recall": 51.3, "support": 1024,
                 "deskripsi": "Mata melebar, mulut terbuka, sering tertukar sedih"},
    "senang":   {"emoji": "😄", "warna": "#22c55e", "recall": 86.5, "support": 1774,
                 "deskripsi": "Senyum lebar, pipi terangkat — kelas paling andal"},
    "netral":   {"emoji": "😐", "warna": "#64748b", "recall": 69.1, "support": 1233,
                 "deskripsi": "Wajah datar, otot rileks"},
    "sedih":    {"emoji": "😢", "warna": "#3b82f6", "recall": 58.7, "support": 1247,
                 "deskripsi": "Sudut bibir turun, alis naik ke tengah"},
    "terkejut": {"emoji": "😯", "warna": "#eab308", "recall": 82.6, "support": 831,
                 "deskripsi": "Mata & mulut terbuka lebar, alis terangkat"},
}

# ═══════════════════════════════════════════════════════════════════
#  CSS — tema modern (gelap, kartu membulat, gradasi halus)
# ═══════════════════════════════════════════════════════════════════
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
    .stApp { font-family: 'Plus Jakarta Sans', system-ui, sans-serif; }
    [data-testid="stMetricValue"] { font-size: 1.9rem; font-weight: 800; }
    [data-testid="stMetricLabel"] { color: #94a3b8; }
    [data-testid="stMetric"] {
        background: linear-gradient(150deg, #1e293b, #0f172a);
        border: 1px solid #334155; border-radius: 16px;
        padding: 14px 18px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px; padding: 8px 18px; font-weight: 600;
    }
    .stTabs [aria-selected="true"] { background: #312e81; }
    .stButton>button, .stDownloadButton>button {
        border-radius: 12px; font-weight: 600;
    }
    div[data-testid="stFileUploader"] { border-radius: 14px; }
    .chip {
        display:inline-block; background:#1e293b; color:#e2e8f0;
        border:1px solid #334155; border-radius:999px;
        padding:4px 12px; margin:2px; font-size:12px;
    }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════
#  KOMPONEN UI
# ═══════════════════════════════════════════════════════════════════
def component_hero(judul, subjudul, badge):
    st.markdown(
        f"""
        <div style="background:linear-gradient(120deg,#0f172a,#1e293b 50%,#312e81);
                    border:1px solid #334155aa;border-radius:22px;
                    padding:26px 32px;color:#f8fafc;margin-bottom:6px">
          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px">
            <div>
              <div style="font-size:30px;font-weight:800;letter-spacing:-0.5px">{judul}</div>
              <div style="color:#94a3b8;margin-top:6px;font-size:14px">{subjudul}</div>
            </div>
            <div style="background:#33415566;border:1px solid #475569;border-radius:999px;
                        padding:10px 18px;font-size:13px;color:#c7d2fe">{badge}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def component_kartu_prediksi(label, keyakinan, probs, nama_kelas, lebar="100%"):
    """Kartu hasil prediksi modern: emoji + label + bar keyakinan + top-3 probabilitas."""
    info = INFO_KELAS.get(label, {"emoji": "🎭", "warna": "#64748b", "deskripsi": ""})
    emoji, warna = info["emoji"], info["warna"]
    teratas = sorted(zip(nama_kelas, probs), key=lambda x: -x[1])[:3]
    baris = ""
    for nm, p in teratas:
        baris += (
            f'<div style="margin-top:8px">'
            f'<div style="display:flex;justify-content:space-between;font-size:12px;color:#94a3b8">'
            f'<span>{nm}</span><span>{p*100:.1f}%</span></div>'
            f'<div style="background:#0f172a;border-radius:8px;height:7px;margin-top:3px;overflow:hidden">'
            f'<div style="background:{warna};width:{max(2,int(p*100))}%;height:7px;border-radius:8px"></div></div></div>'
        )
    return (
        f'<div style="background:linear-gradient(160deg,{warna}1f,#0f172a);'
        f'border:1px solid {warna}55;border-radius:18px;padding:18px 20px;'
        f'color:#f1f5f9;max-width:{lebar}">'
        f'<div style="display:flex;align-items:center;gap:14px">'
        f'<div style="font-size:38px">{emoji}</div>'
        f'<div>'
        f'<div style="font-size:22px;font-weight:800">{label}</div>'
        f'<div style="color:#94a3b8;font-size:13px">'
        f'Keyakinan {keyakinan*100:.1f}% · {info.get("deskripsi", "")}'
        f'</div>'
        f'</div>'
        f'</div>'
        f'<div style="margin-top:12px;background:#0f172a;border-radius:10px;height:10px;overflow:hidden">'
        f'<div style="background:{warna};width:{max(2,int(keyakinan*100))}%;height:10px;border-radius:10px"></div>'
        f'</div>{baris}'
        f'</div>'
    )


def component_metric(label, nilai, delta=None, bantuan=None, help=None):
    st.metric(label=label, value=nilai, delta=delta, help=help or bantuan)


def komponen_chip(teks):
    return f'<span class="chip">{teks}</span>'


# ═══════════════════════════════════════════════════════════════════
#  MUAT MODEL & DETEKTOR WAJAH (di-cache)
# ═══════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Memuat model…")
def muat_model(path):
    return YOLO(path)


@st.cache_resource
def muat_detektor():
    """Detektor wajah berlapis: YuNet DNN → MediaPipe → Haar Cascade (file model dibundel).

    YuNet (OpenCV FaceDetectorYN) jadi prioritas karena paling andal & cepat untuk
    kamera real-time; MediaPipe dipakai bila tersedia; Haar sebagai cadangan.
    Mengembalikan dict {"utama": (jenis, objek) | None, "haar": objek | None, "label": str}.
    """
    folder_app = Path(__file__).parent
    utama, haar, label = None, None, "tidak tersedia"

    # 1) YuNet DNN (OpenCV FaceDetectorYN) — paling andal untuk webcam/live
    onnx = folder_app / "face_detection_yunet_2023mar.onnx"
    if getattr(cv2, "FaceDetectorYN", None) is not None and onnx.exists():
        try:
            obj = cv2.FaceDetectorYN.create(
                str(onnx), "", (320, 320),
                score_threshold=0.6, nms_threshold=0.3, top_k=5000,
                backend_id=cv2.dnn.DNN_BACKEND_OPENCV, target_id=cv2.dnn.DNN_TARGET_CPU,
            )
            utama, label = ("yunet", obj), "YuNet DNN"
        except Exception:
            utama, label = None, "tidak tersedia"

    # 2) MediaPipe Tasks API (kalau binding C berfungsi pada Python ini)
    if utama is None and mp is not None:
        try:
            from mediapipe.tasks import python as mp_python  # type: ignore
            from mediapipe.tasks.python import vision as mp_vision  # type: ignore
            tflite = folder_app / "blaze_face_short_range.tflite"
            if tflite.exists():
                obj = mp_vision.FaceDetector.create_from_options(
                    mp_vision.FaceDetectorOptions(
                        base_options=mp_python.BaseOptions(model_asset_path=str(tflite)),
                        running_mode=mp_vision.RunningMode.IMAGE,
                        min_detection_confidence=0.5,
                    )
                )
                utama, label = ("tasks", obj), "MediaPipe FaceDetector"
        except Exception:
            utama = None

    # 3) MediaPipe legacy (mp.solutions) — hanya bila API lama masih ada
    if utama is None and mp is not None and hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
        try:
            obj = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
            utama, label = ("legacy", obj), "MediaPipe FaceDetection"
        except Exception:
            utama = None

    # 4) Haar Cascade yang dibundel — selalu dimuat sebagai cadangan terakhir
    kandidat = [
        str(folder_app / "haarcascade_frontalface_default.xml"),
        os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades") else "",
    ]
    path_haar = next((p for p in kandidat if p and os.path.exists(p)), None)
    if path_haar:
        c = cv2.CascadeClassifier(path_haar)
        if not c.empty():
            haar = c
            if utama is None:
                label = "Haar Cascade"
    return {"utama": utama, "haar": haar, "label": label}


def prediksi_potongan(bgr):
    if bgr is None or bgr.size == 0:
        return None
    hasil = model(bgr, verbose=False)
    prob = hasil[0].probs
    semua = prob.data.cpu().numpy() if hasattr(prob.data, "cpu") else np.array(prob.data)
    return {"label": model.names[int(prob.top1)],
            "keyakinan": float(prob.top1conf),
            "prob": semua}


def deteksi_wajah_list(bgr):
    t, l, _ = bgr.shape
    wajah = []
    utama = detektor.get("utama")
    if utama is not None:
        jenis, obj = utama
        if jenis == "yunet":  # OpenCV YuNet → kotak piksel langsung
            try:
                obj.setInputSize((l, t))
                _, faces = obj.detect(bgr)
                if faces is not None:
                    for f in faces:
                        x, y, w, h = [int(v) for v in f[:4]]
                        if float(f[-1]) >= 0.6 and w > 0 and h > 0:
                            wajah.append((max(0, x), max(0, y), min(l, x + w), min(t, y + h)))
            except Exception:
                pass
        elif jenis == "tasks":  # MediaPipe Tasks API (kotak piksel)
            try:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                has = obj.detect(img)
                if has and has.detections:
                    for dt in has.detections:
                        bb = dt.bounding_box
                        x1, y1 = bb.origin_x, bb.origin_y
                        x2, y2 = x1 + bb.width, y1 + bb.height
                        wajah.append((max(0, x1), max(0, y1), min(l, x2), min(t, y2)))
            except Exception:
                pass
        elif jenis == "legacy":  # MediaPipe mp.solutions (box relatif)
            try:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                has = obj.process(rgb)
                if has and has.detections:
                    for dt in has.detections:
                        kotak = dt.location_data.relative_bounding_box
                        x1 = int(kotak.xmin * l); y1 = int(kotak.ymin * t)
                        x2 = x1 + int(kotak.width * l); y2 = y1 + int(kotak.height * t)
                        wajah.append((max(0, x1), max(0, y1), min(l, x2), min(t, y2)))
            except Exception:
                pass
    # cadangan: Haar Cascade (dengan equalizeHist agar tahan cahaya webcam)
    if not wajah and detektor.get("haar") is not None:
        abu = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        abu = cv2.equalizeHist(abu)
        daftar = detektor["haar"].detectMultiScale(abu, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50))
        wajah = [(x, y, x + w, y + h) for (x, y, w, h) in daftar]
    # cadangan terakhir: foto dekat berukuran kecil (≤300 px) → pakai full-frame
    if not wajah and max(t, l) <= 300:
        wajah.append((0, 0, l, t))
    return wajah


def anotasi_frame(bgr):
    """Deteksi wajah + prediksi tiap wajah → gambar anotasi + daftar hasil."""
    anotasi = bgr.copy()
    deteksi = []
    for i, (x1, y1, x2, y2) in enumerate(deteksi_wajah_list(bgr)):
        potongan = bgr[y1:y2, x1:x2]
        if potongan.size == 0:
            continue
        r = prediksi_potongan(potongan)
        if r is None:
            continue
        warna = INFO_KELAS.get(r["label"], {}).get("warna", "#22c55e")
        warna_bgr = tuple(int(warna[i:i+2], 16) for i in (5, 3, 1))
        cv2.rectangle(anotasi, (x1, y1), (x2, y2), warna_bgr, 2)
        label = f"{r['label']} {r['keyakinan']:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(anotasi, (x1, max(0, y1 - th - 12)), (x1 + tw + 12, y1), warna_bgr, -1)
        cv2.putText(anotasi, label, (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (10, 10, 15), 2, cv2.LINE_AA)
        deteksi.append({"kotak": (x1, y1, x2, y2), **r})
    return anotasi, deteksi


# ═══════════════════════════════════════════════════════════════════
#  PILIH MODEL + SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan Model")

    if DAFTAR_MODEL:
        label_pilihan = {
            str(p): (
                f"⭐ Model Final — {p.parent.parent.name if p.parent.parent.name != 'models' else 'best.pt'}"
                if p == REPO_ROOT / "models" / "best.pt"
                else f"{p.parent.parent.name} — {METRIK_BY_FILE.get(p.parent.parent.name, ('—', '—'))[0]:.2f}%"
            )
            for p in DAFTAR_MODEL
        }
        pilihan = st.selectbox(
            "Pilih checkpoint model",
            [str(p) for p in DAFTAR_MODEL],
            format_func=lambda p: label_pilihan[p],
        )
        MODEL_PATH = pilihan
    else:
        MODEL_PATH = "models/best.pt"
        st.warning("Model tidak ditemukan. Jalankan notebook export dulu.")

    ambang = st.slider("Batas keyakinan rendah (info)", 0.0, 1.0, 0.5, 0.05)

    st.divider()
    st.markdown("#### 😶 Daftar Kelas & Recall Uji")
    for k in sorted(INFO_KELAS):
        info = INFO_KELAS[k]
        st.markdown(
            f"{info['emoji']} **{k}** — {info['recall']:.1f}% "
            f"<span style='color:#64748b;font-size:12px'>(uji {info['support']})</span>",
            unsafe_allow_html=True,
        )
    st.divider()
    if st.button("🧹 Kosongkan Riwayat Kamera", width="stretch"):
        st.session_state.rekap_kamera = []
        st.rerun()

# ===== muat model & detektor sekali untuk seluruh sesi =====
model = muat_model(MODEL_PATH)
DAFTAR_KELAS = list(model.names.values())
detektor = muat_detektor()

nama_model = Path(MODEL_PATH).parent.parent.name
metrik = METRIK_BY_FILE.get(nama_model) or METRIK_BY_FILE.get(Path(MODEL_PATH).name) or (None, None)

if "rekap_kamera" not in st.session_state:
    st.session_state.rekap_kamera = []
if "kamera_aktif" not in st.session_state:
    st.session_state.kamera_aktif = False

# ═══════════════════════════════════════════════════════════════════
#  HALAMAN UTAMA
# ═══════════════════════════════════════════════════════════════════
component_hero(
    "😊 Klasifikasi Ekspresi Wajah",
    "YOLOv8m-cls · FER2013 · 7 ekspresi · seed 42 (deterministik) — validasi otomatis 7.178 gambar uji asli.",
    f"Model: <b>{Path(MODEL_PATH).parent.parent.name}</b>",
)

m1, m2, m3, m4 = st.columns(4)
component_metric("Top-1 Akurasi (uji)", f"{metrik[0]:.2f}%" if metrik[0] else "—",
                 help="Akurasi label tunggal pada 7.178 gambar uji asli FER2013")
component_metric("Top-5 Akurasi (uji)", f"{metrik[1]:.2f}%" if metrik[1] else "—",
                 help="Label benar masuk 5 prediksi teratas")
component_metric("Jumlah Kelas", f"{len(DAFTAR_KELAS)}", help="marah · jijik · takut · senang · netral · sedih · terkejut")
component_metric("Detektor Wajah", detektor["label"], help="Prioritas: YuNet DNN → MediaPipe → Haar Cascade (file model dibundel di folder aplikasi)")

st.markdown("---")

tab_gambar, tab_kamera, tab_tujuh, tab_video, tab_info = st.tabs(
    ["📤 Analisis Gambar", "🎥 Kamera Langsung", "🧪 Uji 7 Ekspresi", "🎞️ Analisis Video", "📊 Info Model"]
)

# ════════════ TAB 1 — GAMBAR ════════════
with tab_gambar:
    st.markdown("#### 📤 Unggah Foto & Analisis")
    k1, k2 = st.columns([2, 3], gap="large")
    with k1:
        unggah = st.file_uploader(
            "Pilih gambar (JPG/PNG) — bisa berisi banyak wajah",
            type=["jpg", "jpeg", "png"],
        )
        if unggah:
            gambar = Image.open(unggah).convert("RGB")
            st.image(gambar, caption="Masukan", width="stretch")
    with k2:
        if unggah:
            bgr = cv2.cvtColor(np.array(gambar), cv2.COLOR_RGB2BGR)
            with st.spinner("Menganalisis ekspresi…"):
                an, det = anotasi_frame(bgr)
            if not det:
                st.warning(
                    "😕 Tidak ada wajah terdeteksi. Coba foto lebih dekat, pose depan, "
                    "dan pencahayaan terang. (Potongan kecil ≤300px otomatis dianalisis full-frame.)"
                )
            else:
                k_an, k_res = st.columns([3, 2], gap="medium")
                with k_an:
                    st.image(cv2.cvtColor(an, cv2.COLOR_BGR2RGB), caption="Hasil anotasi", width="stretch")
                with k_res:
                    for d in det:
                        st.markdown(
                            component_kartu_prediksi(d["label"], d["keyakinan"], d["prob"], DAFTAR_KELAS),
                            unsafe_allow_html=True,
                        )
                    if len(det) > 1:
                        st.caption(f"👥 {len(det)} wajah terdeteksi.")

# ════════════ TAB 2 — KAMERA REAL-TIME (bounding box live) ════════════
with tab_kamera:
    st.markdown("#### 🎥 Kamera Real-Time — bounding box langsung di video")
    st.caption(
        "Klik **🟢 Mulai Kamera** → kotak pembatas (bounding box) + label ekspresi + persentase "
        "langsung tampil di video secara live. Gunakan pose depan, cahaya terang, jarak ±50 cm. "
        "*(Memakai webcam laptop lewat OpenCV — tidak bergantung izin kamera di browser.)*"
    )

    c_mulai, c_berhenti = st.columns(2)
    with c_mulai:
        if st.button("🟢 Mulai Kamera", type="primary", width="stretch"):
            st.session_state.kamera_aktif = True
    with c_berhenti:
        if st.button("⏹ Berhenti", width="stretch"):
            st.session_state.kamera_aktif = False

    if st.session_state.kamera_aktif:
        # buka webcam (coba index 0 dulu, lalu 1)
        cap = None
        for idx in (0, 1):
            c = cv2.VideoCapture(idx)
            if c.isOpened():
                cap = c
                break
            c.release()
        if cap is None:
            st.session_state.kamera_aktif = False
            st.error(
                "❌ Kamera tidak dapat diakses. Tutup aplikasi lain yang memakai webcam "
                "(browser/meeting), lalu klik Mulai lagi."
            )
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            st.success("🟢 Kamera menyala — bounding box & ekspresi tampil live di video.")
            k_video, k_info = st.columns([3, 2], gap="large")
            with k_video:
                holder_video = st.empty()      # video + bounding box live (ukuran terkontrol)
            with k_info:
                holder_status = st.empty()     # status FPS / jumlah wajah
                holder_info = st.empty()       # kartu ekspresi live
            rekap = st.session_state.rekap_kamera
            t0, hitung, fps = time.time(), 0, 0.0
            try:
                while st.session_state.kamera_aktif:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    an, det = anotasi_frame(frame)
                    hitung += 1
                    now = time.time()
                    if now - t0 >= 1.0:
                        fps = hitung / (now - t0)
                        hitung, t0 = 0, now
                    holder_video.image(cv2.cvtColor(an, cv2.COLOR_BGR2RGB), width=560)
                    if det:
                        d0 = det[0]
                        emo = INFO_KELAS.get(d0["label"], {}).get("emoji", "🙂")
                        rekap.append({"label": d0["label"], "keyakinan": d0["keyakinan"]})
                        if len(rekap) > 1000:
                            del rekap[: len(rekap) - 1000]
                        teks = (
                            f"🟢 LIVE · FPS ±{fps:.0f} · Wajah: {len(det)} · "
                            f"Ekspresi: **{d0['label']}** ({d0['keyakinan'] * 100:.0f}%)"
                        )
                        holder_info.markdown(
                            f"<div style='background:#0f172a;border:1px solid #eab308;border-radius:14px;"
                            f"padding:14px 16px;'><div style='color:#94a3b8;font-size:.85rem'>"
                            f"EKSPRESI TERDETEKSI</div><div style='font-size:1.5rem;font-weight:700;"
                            f"color:#fbbf24'>{emo} {d0['label']}</div>"
                            f"<div style='color:#94a3b8;font-size:.9rem;margin-top:2px'>"
                            f"Keyakinan <b>{d0['keyakinan'] * 100:.0f}%</b></div></div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        teks = (
                            f"🟡 LIVE · FPS ±{fps:.0f} · Wajah belum terdeteksi — "
                            f"dekatkan wajah / perbaiki cahaya"
                        )
                        holder_info.markdown(
                            f"<div style='background:#0f172a;border:1px solid #334155;border-radius:14px;"
                            f"padding:14px 16px;color:#64748b;font-size:.9rem'>Menunggu wajah… "
                            f"<br>Posisikan wajah di depan kamera.</div>",
                            unsafe_allow_html=True,
                        )
                    holder_status.caption(teks)
            finally:
                cap.release()
            st.session_state.kamera_aktif = False
            st.info("⏹ Kamera dihentikan.")
    else:
        st.info("Kamera dalam keadaan mati. Klik **🟢 Mulai Kamera** untuk streaming deteksi real-time.")

    st.divider()
    st.markdown("##### 📈 Rekap Sesi (terkumpul otomatis saat kamera menyala)")
    rekap = st.session_state.rekap_kamera
    if not rekap:
        st.info("Belum ada data. Mulai kamera dan biarkan beberapa detik.")
    else:
        hit = Counter(r["label"] for r in rekap)
        tot = len(rekap)
        df_rc = pd.DataFrame(
            [{"Ekspresi": f"{INFO_KELAS.get(k, {}).get('emoji', '')} {k}",
              "Jumlah": hit.get(k, 0),
              "Persentase": f"{hit.get(k, 0) / tot * 100:.1f}%"} for k in DAFTAR_KELAS]
        )
        st.table(df_rc[df_rc["Jumlah"] > 0] if (df_rc["Jumlah"] > 0).any() else df_rc)
        c3 = st.columns(3)
        c3[0].metric("Jumlah Frame", tot)
        c3[1].metric("Ekspresi Dominan", hit.most_common(1)[0][0] if hit else "—")
        c3[2].metric(
            "Keyakinan Rata-rata",
            f"{np.mean([r['keyakinan'] for r in rekap]) * 100:.1f}%" if rekap else "—",
        )

# ════════════ TAB 3 — UJI 7 EKSPRESI ════════════
with tab_tujuh:
    st.markdown("#### 🧪 Uji 7 Ekspresi — Apakah model mengenali semua ekspresi?")
    st.markdown(
        "Unggah **1 foto per ekspresi** yang Anda peragakan. Sistem memeriksa apakah prediksi = "
        "label yang diharapkan — cara paling jujur menguji model (perhatikan kelas **takut/sedih** "
        "yang memang sulit, recall uji 51% / 59%)."
    )
    st.caption("Tips: pose ekspresif, cahaya depan, jarak ±50 cm, 1 orang per foto, hindari masker/kacamata gelap.")

    kolom = st.columns(7)
    unggahan = {}
    for i, k in enumerate(DAFTAR_KELAS):
        with kolom[i]:
            info = INFO_KELAS.get(k, {"emoji": "🎭"})
            st.markdown(f"### {info['emoji']} {k}")
            up = st.file_uploader(f"Foto {k}", type=["jpg", "jpeg", "png"], key=f"up_{k}", label_visibility="collapsed")
            unggahan[k] = up
            if up:
                st.image(Image.open(up).convert("RGB"), width="stretch")

    if st.button("▶️ Jalankan Validasi 7 Ekspresi", type="primary"):
        if not any(unggahan.values()):
            st.warning("Unggah minimal 1 foto terlebih dahulu.")
        else:
            hasil = []
            for k in DAFTAR_KELAS:
                up = unggahan[k]
                if up is None:
                    hasil.append({"diharapkan": k, "prediksi": "—", "keyakinan": None, "benar": None})
                    continue
                gm = Image.open(up).convert("RGB")
                bgr = cv2.cvtColor(np.array(gm), cv2.COLOR_RGB2BGR)
                an, det = anotasi_frame(bgr)
                if not det:
                    hasil.append({"diharapkan": k, "prediksi": "tidak ada wajah", "keyakinan": 0,
                                  "benar": False, "an": an})
                else:
                    d = det[0]
                    hasil.append({"diharapkan": k, "prediksi": d["label"], "keyakinan": d["keyakinan"],
                                  "benar": d["label"] == k, "an": an, "prob": d["prob"]})

            st.divider()
            jml_benar = sum(1 for r in hasil if r["benar"] is True)
            jml_uji = sum(1 for r in hasil if r["benar"] is not None)
            akurasi = jml_benar / jml_uji * 100 if jml_uji else 0
            component_metric("Akurasi Uji Manual", f"{akurasi:.1f}%", f"{jml_benar}/{jml_uji} benar")

            df = pd.DataFrame([
                {"Diharapkan": r["diharapkan"],
                 "Prediksi": r["prediksi"],
                 "Keyakinan": f"{r['keyakinan'] * 100:.1f}%" if r["keyakinan"] else "—",
                 "Hasil": "✅ BENAR" if r["benar"] is True else ("❌ SALAH" if r["benar"] is False else "—")}
                for r in hasil
            ])
            st.table(df)

            st.markdown("##### Rincian per Ekspresi")
            k2 = st.columns(7)
            for i, r in enumerate(hasil):
                with k2[i]:
                    if r.get("an") is not None:
                        st.image(cv2.cvtColor(r["an"], cv2.COLOR_BGR2RGB),
                                 caption=f"{r['diharapkan']} → {r['prediksi']}", width="stretch")
                    if r["benar"] is False and r.get("prob") is not None:
                        kamus = {model.names[i_]: float(v) for i_, v in enumerate(r["prob"])}
                        st.bar_chart(kamus, height=140, color="#f59e0b")
                        if r["diharapkan"] in ("takut", "sedih") and r["prediksi"] in ("takut", "sedih", "netral"):
                            st.caption("⚠️ Kebingungan alami takut↔sedih↔netral (lihat info model).")

            st.info(
                "Model final (V3 + Fine-Tune) top-1 **69,6%** pada uji asli FER2013. "
                "Kelas **senang 87% & terkejut 83%** hampir selalu benar; **takut 51% & sedih 59%** "
                "paling sering tertukar — wajar untuk data FER grayscale, bukan bug. "
                "Coba pose lebih ekspresif atau bandingkan dengan mode Analisis Gambar."
            )

# ════════════ TAB 4 — VIDEO ════════════
with tab_video:
    st.markdown("#### 🎞️ Analisis Video (sampling per detik)")
    upv = st.file_uploader("Pilih video (mp4/avi/mov/mkv)", type=["mp4", "avi", "mov", "mkv"])
    cuplik = st.slider("Ambil 1 frame per N detik", 1, 5, 1)
    if upv:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(upv.read())
        tfile.close()
        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st.caption(f"FPS {fps:.1f} · total {total} frame (±{total / fps:.1f} detik)")
        if st.button("▶️ Analisis Video", type="primary"):
            prog = st.progress(0, text="Memproses frame…")
            idx, hasil_detik = 0, []
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                langkah = max(1, int(fps * cuplik))
                if idx % langkah == 0:
                    an, det = anotasi_frame(frame)
                    lab = det[0]["label"] if det else "tidak ada wajah"
                    key = det[0]["keyakinan"] if det else 0
                    hasil_detik.append({"detik": idx / fps, "label": lab, "keyakinan": key, "an": an})
                idx += 1
                prog.progress(min(idx / total, 1.0) if total else 0,
                              text=f"Frame {idx}/{total}")
            cap.release()
            os.unlink(tfile.name)
            prog.empty()

            if hasil_detik:
                daftar = [r["label"] for r in hasil_detik if r["label"] != "tidak ada wajah"]
                m_a, m_b = st.columns(2)
                if daftar:
                    hit = Counter(daftar)
                    dom = hit.most_common(1)[0]
                    m_a.metric("Ekspresi Dominan", f"{dom[0]}", f"{dom[1]}/{len(daftar)} frame")
                    m_b.metric("Frame Teranalisis", len(hasil_detik))
                    st.bar_chart({k: hit.get(k, 0) for k in DAFTAR_KELAS})
                else:
                    m_a.metric("Ekspresi Dominan", "—")
                    m_b.metric("Frame Teranalisis", len(hasil_detik))
                    st.warning("Tidak ada wajah terdeteksi di seluruh video.")

                st.markdown("##### Garis Waktu")
                st.dataframe(
                    pd.DataFrame([
                        {"Detik": f"{r['detik']:.1f}", "Ekspresi": r["label"],
                         "Keyakinan": f"{r['keyakinan'] * 100:.1f}%"} for r in hasil_detik
                    ]),
                    width="stretch",
                    hide_index=True,
                )
                with st.expander("🖼️ Lihat frame hasil"):
                    for r in hasil_detik:
                        st.image(cv2.cvtColor(r["an"], cv2.COLOR_BGR2RGB),
                                 caption=f"Detik {r['detik']:.1f} — {r['label']} {r['keyakinan']:.0%}",
                                 width="stretch")
        else:
            cap.release()
            os.unlink(tfile.name)

# ════════════ TAB 5 — INFO MODEL ════════════
with tab_info:
    st.markdown("#### 📊 Info Model & Bukti Kinerja")

    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.markdown("**Model yang sedang dipakai:**")
        st.code(str(MODEL_PATH), language=None)
        df_info = pd.DataFrame([
            {"Properti": "Arsitektur", "Nilai": "YOLOv8m-cls (42 layer · 15.771.623 parameter · 41,6 GFLOPs)"},
            {"Properti": "Tahap 1 — Baseline", "Nilai": "150 epoch, freeze=9, lr0=0.0005 → Top-1 62,93% (best E60)"},
            {"Properti": "Tahap 2 — Fine-Tune", "Nilai": "60 epoch, freeze=4, lr0=0.0002 → Top-1 69,59% (best E47)"},
            {"Properti": "Akurasi Uji (7.178 asli)", "Nilai": f"Top-1 {metrik[0]:.2f}% · Top-5 {metrik[1]:.2f}%" if metrik[0] else "—"},
            {"Properti": "Reproduksibilitas", "Nilai": "seed 42 · deterministic=True · amp=False · cos_lr"},
            {"Properti": "Dataset", "Nilai": "FER2013 — 47.000 train (CLAHE+224+balance) / 7.178 val asli"},
        ])
        st.table(df_info)

        st.markdown("**Rekap Validasi Final — recall per kelas:**")
        df_kelas = pd.DataFrame([
            {"Ekspresi": f"{v['emoji']} {k}", "Recall Uji": f"{v['recall']:.1f}%",
             "Sampel Uji": v["support"], "Catatan": v["deskripsi"]}
            for k, v in sorted(INFO_KELAS.items())
        ])
        st.table(df_kelas)
    with c2:
        cm_path = REPO_ROOT / "docs" / "confusion_matrix_v3.png"
        if cm_path.exists():
            st.markdown("**Confusion matrix — validasi final (computed by notebook):**")
            st.image(Image.open(cm_path), width="stretch")
        else:
            st.info("`docs/confusion_matrix_v3.png` belum ada — jalankan sel Validasi di notebook.")

    kurva_path = REPO_ROOT / "docs" / "results_v3.png"
    if kurva_path.exists():
        st.markdown("**Kurva pelatihan & validasi (Tahap 2 — Fine-Tune):**")
        st.image(Image.open(kurva_path), width="stretch")

    st.markdown(
        "---\n"
        "**Catatan jujur:** kelas **takut (51%)** dan **sedih (59%)** adalah yang paling sulit untuk data "
        "FER2013 grayscale — saling tertukar dengan netral. Semua angka diambil dari validasi otomatis "
        "7.178 gambar uji asli (bukan data latih), tercatat permanen di notebook "
        "`nb-klasifikasi-ekspresi-wajah.ipynb` (log eksekusi asli tiap sel) dan `runs/*/results.csv`."
    )

st.markdown(
    "<div style='margin-top:20px;color:#64748b;font-size:12px;text-align:center'>"
    "YOLOv8m-cls · FER2013 · seed 42 (deterministik) · Top-1 69,59% / Top-5 98,98% — validasi otomatis 7.178 gambar uji asli"
    "</div>",
    unsafe_allow_html=True,
)
