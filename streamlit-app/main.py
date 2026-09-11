"""
Streamlit - Uji Validasi 7 Ekspresi YOLOv8m (224px)
V2 - Label Indonesia: jijik, marah, netral, sedih, senang, takut, terkejut
Fitur: Unggah Gambar, Kamera Langsung, Uji 7 Ekspresi, Unggah Video
Bahasa: Indonesia penuh
"""
import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from collections import Counter
from PIL import Image
import pandas as pd

from ultralytics import YOLO
try:
    import mediapipe as mp
except ImportError:
    mp = None

# ========= PENGATURAN HALAMAN =========
st.set_page_config(page_title="Uji Ekspresi YOLOv8m", page_icon="😊", layout="wide")

# Cari model V2
DAFTAR_MODEL = [
    r"C:\Users\Admin\Documents\Kuliah\herfandi-ml\hasil_ekspresi_yolov8m\yolov8m-ekpresi\weights\best.pt",
    r"C:\Users\Admin\Documents\Kuliah\herfandi-ml\facial-expression-classification\models\best.pt",
    "yolov8m-ekpresi/weights/best.pt",
    "weights/best.pt",
    "best.pt",
]
MODEL_PATH = next((p for p in DAFTAR_MODEL if os.path.exists(p)), DAFTAR_MODEL[0])

INFO_KELAS = {
    'marah':    {'emoji':'😠','warna':'#dc3545','deskripsi':'Marah — alis mengerut, rahang mengencang'},
    'jijik':    {'emoji':'🤢','warna':'#6f42c1','deskripsi':'Jijik — hidung mengerut, bibir atas terangkat'},
    'takut':    {'emoji':'😨','warna':'#fd7e14','deskripsi':'Takut — mata melebar, mulut terbuka'},
    'senang':   {'emoji':'😄','warna':'#28a745','deskripsi':'Senang — senyum, pipi terangkat'},
    'netral':   {'emoji':'😐','warna':'#6c757d','deskripsi':'Netral — wajah datar'},
    'sedih':    {'emoji':'😢','warna':'#007bff','deskripsi':'Sedih — sudut bibir turun'},
    'terkejut': {'emoji':'😯','warna':'#ffc107','deskripsi':'Terkejut — mata & mulut terbuka lebar'},
}

if 'basis_data_lacak' not in st.session_state:
    st.session_state.basis_data_lacak = {}

# ========= MUAT MODEL =========
@st.cache_resource
def muat_model(path):
    if not os.path.exists(path):
        st.error(f"Model tidak ditemukan: {path}")
        st.stop()
    m = YOLO(path)
    return m

model = muat_model(MODEL_PATH)
# Ambil daftar kelas langsung dari model (urutan: jijik, marah, netral, sedih, senang, takut, terkejut)
DAFTAR_KELAS = list(model.names.values())
PEMETAAN_NAMA_KE_ID = {v:k for k,v in model.names.items()}

# Detektor wajah
deteksi_wajah = None
haar_cascade = None
if mp is not None and hasattr(mp, "solutions"):
    try:
        if hasattr(mp.solutions, "face_detection"):
            mp_wajah = mp.solutions.face_detection
            deteksi_wajah = mp_wajah.FaceDetection(min_detection_confidence=0.5)
    except Exception:
        deteksi_wajah = None
if deteksi_wajah is None:
    kandidat = [
        os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml") if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades") else "",
        os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml"),
        "haarcascade_frontalface_default.xml",
    ]
    path_haar = next((p for p in kandidat if p and os.path.exists(p)), None)
    if path_haar:
        haar_cascade = cv2.CascadeClassifier(path_haar)

def prediksi_potongan(bgr):
    if bgr is None or bgr.size == 0:
        return None, None, None
    hasil = model(bgr, verbose=False)
    prob = hasil[0].probs
    idx = int(prob.top1)
    keyakinan = float(prob.top1conf)
    label = model.names[idx]
    semua_prob = prob.data.cpu().numpy() if hasattr(prob.data, 'cpu') else np.array(prob.data)
    return label, keyakinan, semua_prob

def deteksi_wajah_list(bgr):
    t,l,_ = bgr.shape
    wajah=[]
    if deteksi_wajah is not None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        hasil = deteksi_wajah.process(rgb)
        if hasil and hasil.detections:
            for det in hasil.detections:
                kotak = det.location_data.relative_bounding_box
                x1=int(kotak.xmin*l); y1=int(kotak.ymin*t)
                x2=x1+int(kotak.width*l); y2=y1+int(kotak.height*t)
                wajah.append((max(0,x1),max(0,y1),min(l,x2),min(t,y2)))
    elif haar_cascade is not None:
        abu=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
        daftar=haar_cascade.detectMultiScale(abu, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
        for (x,y,w,h) in daftar:
            wajah.append((x,y,x+w,y+h))
    if not wajah and max(t,l) <= 300:
        wajah=[(0,0,l,t)]
    return wajah

def anotasi_frame(bgr):
    daftar_wajah=deteksi_wajah_list(bgr)
    anotasi=bgr.copy()
    deteksi=[]
    for i,(x1,y1,x2,y2) in enumerate(daftar_wajah):
        potongan=bgr[y1:y2, x1:x2]
        if potongan.size==0: continue
        label,keyakinan,prob=prediksi_potongan(potongan)
        if label is None: continue
        id_lacak=f"Wajah_{i+1}"
        if id_lacak not in st.session_state.basis_data_lacak:
            st.session_state.basis_data_lacak[id_lacak]=[]
        st.session_state.basis_data_lacak[id_lacak].append(label)
        warna=(0,255,0)
        cv2.rectangle(anotasi,(x1,y1),(x2,y2),warna,2)
        cv2.putText(anotasi,f"{label} {keyakinan:.0%}",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,warna,2)
        deteksi.append({"kotak":(x1,y1,x2,y2),"label":label,"keyakinan":keyakinan,"prob":prob})
    return anotasi, deteksi

# ========= JUDUL =========
st.title("😊 Uji Validasi Ekspresi — YOLOv8m V2 (71,2% FER2013)")
st.caption(f"Model: `{os.path.basename(MODEL_PATH)}` | Kelas: {', '.join(DAFTAR_KELAS)} | ukuran: 224 | {model.names}")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Pengaturan")
    ambang = st.slider("Ambang keyakinan (hanya info)", 0.0, 1.0, 0.5)
    st.write(f"Lokasi model: `{MODEL_PATH}`")
    st.write(f"Perangkat: otomatis")
    st.divider()
    st.header("📋 Cara Uji 7 Ekspresi")
    st.info("1. Pilih mode **🧪 Uji 7 Ekspresi**\n2. Unggah 1 foto tiap ekspresi (pose depan, cahaya terang)\n3. Lihat ✅/❌ apakah prediksi = ekspresi yang diharapkan\n4. Atau pakai **Unggah Gambar** untuk tes bebas\n5. **Kamera Langsung** untuk tes langsung")
    if st.button("🔄 Atur Ulang Log"):
        st.session_state.basis_data_lacak={}
        st.rerun()
    st.divider()
    st.header("😶 Daftar Kelas")
    for k in DAFTAR_KELAS:
        info=INFO_KELAS.get(k, {"emoji":"•","deskripsi":k})
        st.write(f"{info['emoji']} **{k}** — {info.get('deskripsi','')}")

mode = st.radio("Pilih mode:", ["📷 Unggah Gambar (bebas)","🎥 Kamera Langsung","🧪 Uji 7 Ekspresi (validasi)","🎞️ Unggah Video"], horizontal=True)

# ===== MODE 1: Unggah bebas =====
if mode == "📷 Unggah Gambar (bebas)":
    k1,k2=st.columns(2)
    with k1:
        st.header("📤 Unggah")
        unggah=st.file_uploader("Pilih gambar JPG/PNG (bisa banyak wajah)", type=['jpg','jpeg','png'])
        if unggah:
            gambar=Image.open(unggah).convert("RGB")
            st.image(gambar, caption="Gambar masukan", use_container_width=True)
    with k2:
        st.header("🔍 Hasil")
        if unggah:
            bgr=cv2.cvtColor(np.array(gambar), cv2.COLOR_RGB2BGR)
            with st.spinner("Sedang menganalisis..."):
                an,det=anotasi_frame(bgr)
            if not det:
                st.warning("Tidak ada wajah terdeteksi. Coba foto lebih dekat/terang. Jika potongan kecil 48px, fallback full-frame aktif.")
            else:
                st.image(cv2.cvtColor(an,cv2.COLOR_BGR2RGB), caption="Hasil", use_container_width=True)
                for d in det:
                    info=INFO_KELAS.get(d['label'], {"emoji":""})
                    st.markdown(f"### {info['emoji']} **{d['label']}** — {d['keyakinan']:.2%} {'✅' if d['keyakinan']>=ambang else '⚠️ keyakinan rendah'}")
                    st.progress(int(d['keyakinan']*100))
                    kamus_prob={model.names[i]: float(v) for i,v in enumerate(d['prob'])}
                    kamus_prob=dict(sorted(kamus_prob.items(), key=lambda x: x[1], reverse=True))
                    st.bar_chart(kamus_prob)
                    if d['label'] in ['takut','sedih','marah','netral']:
                        st.caption("⚠️ Kelas ini sering tertukar (lihat matriks kebingungan: takut↔sedih, sedih↔netral).")

# ===== MODE 2: Kamera =====
elif mode == "🎥 Kamera Langsung":
    st.header("🎥 Kamera Langsung")
    jalan=st.checkbox("Aktifkan Kamera", value=False)
    kol1,kol2=st.columns([2,1])
    jendela=kol1.image([])
    area_stat=kol2.empty()
    if jalan:
        cap=cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Kamera tidak tersedia.")
        else:
            while jalan:
                ret,frame=cap.read()
                if not ret: break
                an,_=anotasi_frame(frame)
                jendela.image(cv2.cvtColor(an,cv2.COLOR_BGR2RGB))
                with area_stat.container():
                    st.markdown("### 📈 Rekap")
                    if not st.session_state.basis_data_lacak:
                        st.write("Belum ada data.")
                    else:
                        for id_lacak,riwayat in st.session_state.basis_data_lacak.items():
                            hit=Counter(riwayat); tot=len(riwayat)
                            st.write(f"**{id_lacak}** ({tot} frame)")
                            df=pd.DataFrame([{"Ekspresi":f"{INFO_KELAS.get(k,{}).get('emoji','')} {k}","Jumlah":hit.get(k,0),"Persentase":f"{hit.get(k,0)/tot*100:.1f}%"} for k in DAFTAR_KELAS])
                            st.table(df)
                            st.bar_chart({k:hit.get(k,0) for k in DAFTAR_KELAS})
            cap.release()
    else:
        st.info("Centang Aktifkan Kamera. Pastikan browser memberi izin kamera.")

# ===== MODE 3: Uji 7 Ekspresi =====
elif mode == "🧪 Uji 7 Ekspresi (validasi)":
    st.header("🧪 Uji 7 Ekspresi — Apakah semua sesuai?")
    st.write("Unggah **1 foto per ekspresi** yang kamu peragakan. Sistem akan cek apakah prediksi = label yang diharapkan. Ini cara paling jujur untuk cek overfit/takut↔sedih.")
    st.caption("Tips: pose ekspresif, cahaya depan, jarak 50cm, 1 orang per foto, hindari masker/kacamata gelap.")

    kolom=st.columns(7)
    unggahan={}
    for i,k in enumerate(DAFTAR_KELAS):
        with kolom[i]:
            info=INFO_KELAS.get(k, {"emoji":""})
            st.markdown(f"**{info['emoji']} {k}**")
            up=st.file_uploader(f"{k}", type=['jpg','jpeg','png'], key=f"up_{k}", label_visibility="collapsed")
            unggahan[k]=up
            if up:
                gm=Image.open(up).convert("RGB")
                st.image(gm, use_container_width=True)

    if st.button("▶️ Jalankan Validasi 7 Ekspresi", type="primary"):
        if not any(unggahan.values()):
            st.warning("Unggah minimal 1 foto dulu.")
        else:
            hasil=[]
            for k in DAFTAR_KELAS:
                up=unggahan[k]
                if up is None:
                    hasil.append({"diharapkan":k,"prediksi":"-","keyakinan":0,"benar":None,"detail":"tidak diunggah"})
                    continue
                gm=Image.open(up).convert("RGB")
                bgr=cv2.cvtColor(np.array(gm), cv2.COLOR_RGB2BGR)
                an,det=anotasi_frame(bgr)
                if not det:
                    hasil.append({"diharapkan":k,"prediksi":"tidak ada wajah","keyakinan":0,"benar":False,"detail":"wajah tidak terdeteksi","an":an})
                else:
                    d=det[0]
                    benar=(d['label']==k)
                    hasil.append({"diharapkan":k,"prediksi":d['label'],"keyakinan":d['keyakinan'],"benar":benar,"prob":d['prob'],"an":an,"det":det})

            st.divider()
            st.subheader("Hasil Validasi")
            jml_benar=sum(1 for r in hasil if r['benar']==True)
            jml_uji=sum(1 for r in hasil if r['benar'] is not None)
            akur=jml_benar/jml_uji*100 if jml_uji else 0
            st.metric(f"Akurasi Manual (dari {jml_uji} foto)", f"{akur:.1f}%", f"{jml_benar}/{jml_uji} benar")

            df=pd.DataFrame([{
                "Diharapkan":r['diharapkan'],
                "Prediksi":r['prediksi'],
                "Keyakinan":f"{r['keyakinan']:.2%}" if r['keyakinan'] else "-",
                "Hasil":"✅ BENAR" if r['benar']==True else ("❌ SALAH" if r['benar']==False else "—"),
                "Top-2 probabilitas": ", ".join([f"{model.names[i]}:{float(v):.2f}" for i,v in sorted(enumerate(r['prob']), key=lambda x: x[1], reverse=True)[:2]]) if 'prob' in r else r.get('detail','')
            } for r in hasil])
            st.table(df)

            st.markdown("#### Rincian per Ekspresi")
            kol2=st.columns(7)
            for i,k in enumerate(DAFTAR_KELAS):
                r=hasil[i]
                with kol2[i]:
                    if 'an' in r:
                        st.image(cv2.cvtColor(r['an'],cv2.COLOR_BGR2RGB), caption=f"{k} -> {r['prediksi']} {r['keyakinan']:.0%}" if r['prediksi']!='-' else k, use_container_width=True)
                    if r['benar']==False:
                        kamus_prob={model.names[idx]:float(v) for idx,v in enumerate(r['prob'])}
                        st.bar_chart(kamus_prob)
                        if k in ["takut","sedih"] and r['prediksi'] in ["takut","sedih","netral"]:
                            st.caption("⚠️ Memang kebingungan tinggi takut↔sedih↔netral (F1 takut 56%). Coba pose lebih ekspresif.")

            st.info("Catatan: Model V2 akurasi uji 71%, kelas **senang 88% & terkejut 82%** hampir selalu benar, tapi **takut 52% & sedih 59%** sering salah. Jika uji manual juga gagal di 2 kelas itu, itu wajar — bukan bug, memang data FER sulit. Coba ganti pose lebih jelas atau latih ulang dengan data tambahan.")
            st.bar_chart({r['diharapkan']: 1 if r['benar'] else 0 for r in hasil if r['benar'] is not None})

    st.divider()
    with st.expander("Atau: Validasi otomatis dari folder data uji (tanpa unggah manual)"):
        st.write("Ambil 2 sampel acak per kelas dari `dataset_fer2013/val` untuk lihat apakah model sudah benar (cek cepat).")
        if st.button("🎲 Ambil Sampel Acak Uji"):
            basis=r"C:\Users\Admin\Documents\Kuliah\herfandi-ml\hasil_ekspresi_yolov8m\dataset_fer2013\val"
            if not os.path.exists(basis):
                basis="dataset_fer2013/val"
            if not os.path.exists(basis):
                st.error(f"Folder uji tidak ditemukan: {basis}")
            else:
                import random, glob
                sampel=[]
                for k in DAFTAR_KELAS:
                    folder=os.path.join(basis,k)
                    if not os.path.exists(folder): continue
                    daftar=glob.glob(os.path.join(folder,"*.*"))
                    if not daftar: continue
                    pilih=random.sample(daftar, min(2,len(daftar)))
                    for pf in pilih:
                        bgr=cv2.imread(pf)
                        if bgr is None: continue
                        label,keyakinan,_=prediksi_potongan(bgr)
                        sampel.append((k, os.path.basename(pf), label, keyakinan, pf, label==k))
                df2=pd.DataFrame([{"Folder (asli)":s[0],"Berkas":s[1],"Prediksi":s[2],"Keyakinan":f"{s[3]:.2%}","Benar?": "✅" if s[5] else "❌"} for s in sampel])
                st.table(df2)
                ak2=sum(1 for s in sampel if s[5])/len(sampel)*100 if sampel else 0
                st.metric("Akurasi sampel acak", f"{ak2:.1f}%", f"{sum(1 for s in sampel if s[5])}/{len(sampel)}")
                kol3=st.columns(3)
                for i,s in enumerate(sampel):
                    with kol3[i%3]:
                        bgr=cv2.imread(s[4])
                        if bgr is not None:
                            st.image(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB), caption=f"{s[0]} -> {s[2]} {s[3]:.0%} {'✅' if s[5] else '❌'}", use_container_width=True)

# ===== MODE 4: Video =====
else:
    st.header("🎞️ Unggah Video")
    upv=st.file_uploader("Pilih video mp4/avi/mov", type=['mp4','avi','mov','mkv'])
    cuplik=st.slider("Ambil 1 frame per N detik", 1, 5, 1)
    if upv:
        tfile=tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(upv.read()); tfile.close()
        cap=cv2.VideoCapture(tfile.name)
        fps=cap.get(cv2.CAP_PROP_FPS) or 30
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st.write(f"FPS {fps:.1f} total {total} frame (~{total/fps:.1f} detik)")
        if st.button("▶️ Analisis"):
            prog=st.progress(0)
            idx=0; hasil_detik=[]
            cap.set(cv2.CAP_PROP_POS_FRAMES,0)
            while True:
                ret,frame=cap.read()
                if not ret: break
                langkah=max(1,int(fps*cuplik))
                if idx%langkah==0:
                    an,det=anotasi_frame(frame)
                    lab=det[0]['label'] if det else 'tidak ada wajah'
                    key=det[0]['keyakinan'] if det else 0
                    hasil_detik.append({"detik":idx/fps,"label":lab,"keyakinan":key,"an":an})
                    with st.expander(f"Detik {idx/fps:.1f} — {lab} {key:.0%}" if det else f"Detik {idx/fps:.1f} — tidak ada wajah"):
                        st.image(cv2.cvtColor(an,cv2.COLOR_BGR2RGB), use_container_width=True)
                idx+=1
                prog.progress(min(idx/total,1.0) if total else 0)
            cap.release(); os.unlink(tfile.name)
            if hasil_detik:
                daftar=[r['label'] for r in hasil_detik if r['label']!='tidak ada wajah']
                if daftar:
                    hit=Counter(daftar); dom=hit.most_common(1)[0]
                    st.success(f"Dominan: {dom[0]} ({dom[1]}/{len(daftar)} = {dom[1]/len(daftar):.0%})")
                    st.bar_chart({k:hit.get(k,0) for k in DAFTAR_KELAS})
                    st.table(pd.DataFrame([{"Detik":f"{r['detik']:.1f} dtk","Label":r['label'],"Keyakinan":f"{r['keyakinan']:.2%}"} for r in hasil_detik]))
        else:
            cap.release(); os.unlink(tfile.name)

st.markdown("---")
st.caption("V2 YOLOv8m 224px • Top1 71,22% • Takut & Sedih memang lemah (56-59% F1). Uji 7 pose ekspresif untuk validasi paling jujur.")
