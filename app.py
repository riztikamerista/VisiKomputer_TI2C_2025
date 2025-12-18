import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
from PIL import Image
import time

# Konfigurasi halaman dengan tema custom
st.set_page_config(
    page_title="Sistem Deteksi Ikan Cupang Menggunakan YOLOv8",
    page_icon="🐠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling
st.markdown("""
    <style>
    /* Background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Card styling */
    .css-1d391kg, .css-12oz5g7 {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    /* Judul styling */
    h1 {
        color: white !important;
        text-align: center;
        font-size: 3.5em !important;
        font-weight: 800 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 10px !important;
    }
    
    /* Subjudul */
    h2, h3 {
        color: #2d3748 !important;
        font-weight: 700 !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Upload box */
    [data-testid="stFileUploader"] {
        background: white;
        border-radius: 15px;
        padding: 30px;
        border: 3px dashed #667eea;
        text-align: center;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        padding: 15px 40px;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 15px 0 rgba(102, 126, 234, 0.5);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(102, 126, 234, 0.7);
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 15px;
        border-left: 5px solid #667eea;
    }
    
    /* Radio buttons */
    .stRadio > label {
        font-size: 1.2em !important;
        font-weight: 600 !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2em !important;
        color: #2d3748 !important;
        font-weight: 800 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: white !important;
        font-size: 1.1em !important;
        font-weight: 600 !important;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Header dengan emoji dan deskripsi
st.markdown("<h1>🐠 Sistem Deteksi Ikan Cupang Menggunakan YOLOv8</h1>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; color: white; font-size: 1.2em; margin-bottom: 30px; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);'>
        <b>Sistem Deteksi Otomatis Menggunakan YOLOv8</b><br>
        Upload gambar atau video ikan cupang untuk deteksi real-time! 🚀
    </div>
""", unsafe_allow_html=True)

# Load Model dengan progress bar
@st.cache_resource
def load_model():
    return YOLO("best.pt")

with st.spinner('🔄 Memuat model AI...'):
    model = load_model()
    time.sleep(0.5)

# Sidebar dengan styling
st.sidebar.markdown("### 🎯 Panel Kontrol")
st.sidebar.markdown("---")

mode = st.sidebar.radio(
    "📂 Pilih Mode Input:",
    ("🖼️ Gambar", "🎥 Video"),
    help="Pilih jenis media yang ingin Anda deteksi"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Pengaturan")
confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.25,
    step=0.05,
    help="Semakin tinggi nilai, semakin strict deteksinya"
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tips:** Gunakan gambar/video dengan pencahayaan yang baik untuk hasil optimal!")

# Main content area
if mode == "🖼️ Gambar":
    st.markdown("## 📸 Deteksi pada Media Gambar")
    st.markdown("---")
    
    col_upload, col_info = st.columns([2, 1])
    
    with col_upload:
        uploaded_file = st.file_uploader(
            "Drag & Drop atau Klik untuk Upload",
            type=['jpg', 'jpeg', 'png'],
            help="Format yang didukung: JPG, JPEG, PNG"
        )
    
    with col_info:
        st.markdown("""
        <div style='background: white; padding: 20px; border-radius: 15px; margin-top: 10px;'>
            <h4 style='color: #667eea; margin-bottom: 15px;'>📋 Informasi</h4>
            <ul style='color: #333;'>
                <li>Format: JPG, PNG</li>
                <li>Max Size: 200MB</li>
                <li>Resolusi: Bebas</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    if uploaded_file is not None:
        st.markdown("---")
        st.markdown("### 🔍 Hasil Analisis")
        
        # Progress bar untuk efek loading
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(100):
            progress_bar.progress(i + 1)
            if i < 30:
                status_text.text("⏳ Memuat gambar...")
            elif i < 70:
                status_text.text("🤖 Memproses AI...")
            else:
                status_text.text("✨ Menyelesaikan...")
            time.sleep(0.01)
        
        progress_bar.empty()
        status_text.empty()
        
        image = Image.open(uploaded_file)
        results = model.predict(image, conf=confidence)
        
        # Hitung jumlah deteksi
        num_detections = len(results[0].boxes)
        
        # Tampilkan metrics
        metric_cols = st.columns(3)
        with metric_cols[0]:
            st.metric("🎯 Objek Terdeteksi", num_detections)
        with metric_cols[1]:
            st.metric("📊 Confidence", f"{confidence*100:.0f}%")
        with metric_cols[2]:
            st.metric("✅ Status", "Selesai")
        
        st.markdown("---")
        
        # Tampilkan gambar dengan ukuran sedang
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📥 Gambar Original")
            st.image(image, use_container_width=False, width=400)
        with col2:
            st.markdown("#### 🎨 Hasil Deteksi")
            st.image(results[0].plot(), use_container_width=False, width=400)
        
        if num_detections > 0:
            st.markdown(f"""
                <div style='background: white; padding: 20px; border-radius: 15px; 
                            border-left: 5px solid #10b981; margin: 20px 0;'>
                    <p style='color: #059669; font-size: 1.3em; font-weight: 700; margin: 0;'>
                        🎉 Berhasil mendeteksi {num_detections} ikan cupang!
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style='background: white; padding: 20px; border-radius: 15px; 
                            border-left: 5px solid #f59e0b; margin: 20px 0;'>
                    <p style='color: #d97706; font-size: 1.3em; font-weight: 700; margin: 0;'>
                        ⚠️ Tidak ada ikan cupang yang terdeteksi. Coba adjust confidence threshold.
                    </p>
                </div>
            """, unsafe_allow_html=True)

elif mode == "🎥 Video":
    st.markdown("## 🎬 Deteksi pada Media Video")
    st.markdown("---")
    
    col_upload, col_info = st.columns([2, 1])
    
    with col_upload:
        uploaded_video = st.file_uploader(
            "Drag & Drop atau Klik untuk Upload Video",
            type=['mp4', 'avi', 'mov'],
            help="Format yang didukung: MP4, AVI, MOV"
        )
    
    with col_info:
        st.markdown("""
        <div style='background: white; padding: 20px; border-radius: 15px; margin-top: 10px;'>
            <h4 style='color: #667eea; margin-bottom: 15px;'>📋 Informasi</h4>
            <ul style='color: #333;'>
                <li>Format: MP4, AVI, MOV</li>
                <li>Max Size: 500MB</li>
                <li>Durasi: Bebas</li>
                <li>FPS: Otomatis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    if uploaded_video is not None:
        st.markdown("---")
        st.markdown("### 🎥 Video Processing")
        
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Display video info
        info_cols = st.columns(3)
        with info_cols[0]:
            st.metric("📹 Total Frame", total_frames)
        with info_cols[1]:
            st.metric("⚡ FPS", fps)
        with info_cols[2]:
            st.metric("⏱️ Durasi", f"{total_frames/fps:.1f}s")
        
        st.markdown("---")
        
        # Frame display dengan ukuran sedang
        col_video, col_spacer = st.columns([2, 1])
        
        with col_video:
            st_frame = st.empty()
            frame_counter = st.empty()
        
        current_frame = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Proses deteksi tiap frame
            results = model.predict(frame, conf=confidence)
            annotated_frame = results[0].plot()
            
            # Tampilkan ke web dengan ukuran dibatasi
            with col_video:
                st_frame.image(annotated_frame, channels="BGR", width=600)
            
            current_frame += 1
            frame_counter.markdown(f"**Frame: {current_frame}/{total_frames}** ({current_frame/total_frames*100:.1f}%)")
            
        cap.release()
        st.success("✅ Video processing selesai!")
        st.balloons()

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: white; padding: 20px; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);'>
        <p style='font-size: 0.9em;'>Powered by <b>YOLOv8</b> & <b>Streamlit</b> 🚀</p>
        <p style='font-size: 0.8em;'>© 2024 - Betta Fish Detection System</p>
    </div>
""", unsafe_allow_html=True)