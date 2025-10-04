import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import time
from collections import deque
import plotly.graph_objects as go

# --- Page setup ---
st.set_page_config(
    page_title="Vibe Check by HKW", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for better styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        background-color: #0f172a; /* dark navy base */
        color: #e2e8f0; /* light gray text */
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00b4d8 0%, #0077b6 100%); /* electric blue gradient */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 2rem;
    }
    
    .profile-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%); /* dark slate gradient */
        padding: 2rem;
        border-radius: 15px;
        color: #f1f5f9;
        margin-bottom: 2rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        border: 1px solid #475569;
    }
    
    .profile-name {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #38bdf8; /* cyan highlight */
    }
    
    .profile-title {
        font-size: 1.1rem;
        font-weight: 300;
        margin-bottom: 1rem;
        color: #94a3b8;
        opacity: 0.9;
    }
    
    .social-links {
        display: flex;
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .social-link {
        background: rgba(51, 65, 85, 0.7);  /* translucent dark blue-gray */
        padding: 0.5rem 1rem;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.4);
        text-decoration: none;
        color: #e2e8f0 !important;
        transition: all 0.3s;
        font-weight: 600;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }

    .social-link:hover {
        background: rgba(59, 130, 246, 0.3);  /* subtle blue on hover */
        border: 1px solid rgba(59, 130, 246, 0.5);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(59, 130, 246, 0.3);
    }
    
    .stButton>button {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        border-radius: 10px;
        transition: all 0.3s;
        background: linear-gradient(135deg, #0077b6 0%, #00b4d8 100%);
        color: white;
        border: none;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #00b4d8 0%, #48cae4 100%);
        transform: translateY(-1px);
    }

    .metric-card {
        background: #1e293b;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #00b4d8;
        color: #f8fafc;
    }
</style>

""", unsafe_allow_html=True)

# --- Model Loading (Cached) ---
@st.cache_resource
def load_model():
    """Load the TensorFlow/Keras model once."""
    try:
        model = tf.keras.models.load_model(
            "models/vgg16_conservative_img160_dr0.28_lr0.0001_best.h5"
        )
        # Warm up the model
        dummy_input = np.zeros((1, 160, 160, 3), dtype=np.float32)
        model.predict(dummy_input, verbose=0)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

if model is None:
    st.stop()

# --- Constants ---
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
EMOTION_COLORS = {
    "Angry": (0, 0, 255),
    "Disgust": (0, 255, 255),
    "Fear": (255, 0, 255),
    "Happy": (0, 255, 0),
    "Sad": (255, 0, 0),
    "Surprise": (255, 255, 0),
    "Neutral": (200, 200, 200)
}

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    st.error("Error loading face cascade.")
    st.stop()

# --- Session State Initialization ---
if 'run_camera' not in st.session_state:
    st.session_state.run_camera = False
if 'last_faces' not in st.session_state:
    st.session_state.last_faces = []

# --- Profile Section ---
st.markdown('<h1 class="main-header">🎭 Vibe Check</h1>', unsafe_allow_html=True)

# Personal Info 
st.markdown("""
<div class="profile-card">
    <div class="profile-name">Hey👋, I'm Htet Khaing Win</div>
    <div class="profile-title">AI/ML nerd</div>
    <p style="margin-bottom: 1rem;">
        Just a nerd trying to be better!
    </p>
    <div class="social-links">
        <a href="https://www.linkedin.com/in/htet-khaing-win/" target="_blank" class="social-link">🔗 LinkedIn</a>
        <a href="https://github.com/htet-khaing-win" target="_blank" class="social-link">💻 GitHub</a>
        <a href="mailto:htetkhaingwinb@gmail.com" class="social-link">📧 Email</a>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Mode Selection ---
st.markdown("### Choose your mode")
mode = st.radio(
    "",
    ["📸 Upload Image", "🎥 Live Camera"],
    horizontal=True,
    label_visibility="collapsed"
)

# --- Prediction Functions ---
@st.cache_data(ttl=0.5)
def predict_emotion_cached(face_bytes):
    """Cached prediction to avoid redundant processing"""
    face_gray = np.frombuffer(face_bytes, dtype=np.uint8).reshape((160, 160))
    face = face_gray.astype("float32") / 255.0
    face_rgb = np.stack([face, face, face], axis=-1)
    face_rgb = np.expand_dims(face_rgb, axis=0)
    predictions = model.predict(face_rgb, verbose=0)[0]
    return predictions

def predict_emotion(face_gray):
    """Predict emotion from a grayscale face image"""
    face_resized = cv2.resize(face_gray, (160, 160))
    face_bytes = face_resized.tobytes()
    predictions = predict_emotion_cached(face_bytes)
    return predictions

def plot_probabilities(predictions):
    """Plot emotion probabilities using Plotly"""
    fig = go.Figure(data=[
        go.Bar(
            x=EMOTIONS,
            y=predictions,
            marker=dict(
                color=predictions,
                colorscale='Viridis',
                showscale=False
            ),
            text=[f'{p:.2%}' for p in predictions],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Emotion Probability Distribution",
        xaxis_title="Emotion",
        yaxis_title="Confidence",
        yaxis=dict(range=[0, 1]),
        height=400,
        font=dict(family="Poppins", size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    st.plotly_chart(fig, use_container_width=True)

# --- IMAGE UPLOAD MODE ---
if mode == "📸 Upload Image":
    st.markdown("---")
    st.subheader("📤 Upload your Image")
    
    uploaded_file = st.file_uploader(
        "Choose a face image",
        type=["jpg", "jpeg", "png"],
        help="Upload an image containing faces to detect emotions"
    )
    
    if uploaded_file is not None:
        # Load and process image
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        if len(faces) == 0:
            st.warning(" No faces detected in the image. Please try another image.")
        else:
            st.success(f" Detected {len(faces)} face(s)")
            
            # Process each face
            results = []
            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                preds = predict_emotion(face)
                emotion = EMOTIONS[np.argmax(preds)]
                confidence = np.max(preds)
                
                # Draw on image
                color = EMOTION_COLORS.get(emotion, (0, 255, 0))
                cv2.rectangle(img_array, (x, y), (x+w, y+h), color, 3)
                
                # Add label with background
                label = f"{emotion}: {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                cv2.rectangle(img_array, (x, y - label_size[1] - 15), 
                            (x + label_size[0] + 10, y), color, -1)
                cv2.putText(img_array, label, (x + 5, y - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
                
                results.append({
                    'emotion': emotion,
                    'confidence': confidence,
                    'predictions': preds
                })
            
            # Display results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(img_array, caption="Detected Faces", use_container_width=True)
            
            with col2:
                for i, result in enumerate(results):
                    st.markdown(f"#### Face {i+1}")
                    st.metric("Emotion", result['emotion'])
                    st.metric("Confidence", f"{result['confidence']:.2%}")
                    st.markdown("---")
            
            # Show probability distribution for first face
            if results:
                st.markdown("### 📊 Detailed Analysis (Face 1)")
                plot_probabilities(results[0]['predictions'])

# --- LIVE CAMERA MODE ---
else:
    st.markdown("---")
    
    def process_frame(frame, frame_count, process_interval=10):
        """Process a single frame with optimized detection and prediction"""
        scale = 0.5
        small_frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        if frame_count % process_interval == 0:
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.3, 
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            st.session_state.last_faces = []
            
            for (x, y, w, h) in faces:
                x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
                face_roi = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                
                try:
                    predictions = predict_emotion(face_roi)
                    emotion = EMOTIONS[np.argmax(predictions)]
                    confidence = np.max(predictions)
                    
                    st.session_state.last_faces.append({
                        'bbox': (x, y, w, h),
                        'emotion': emotion,
                        'confidence': confidence
                    })
                except Exception as e:
                    continue
        
        for face_data in st.session_state.last_faces:
            x, y, w, h = face_data['bbox']
            emotion = face_data['emotion']
            confidence = face_data['confidence']
            color = EMOTION_COLORS.get(emotion, (255, 255, 255))
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            label = f"{emotion}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            cv2.putText(frame, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return frame

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("📹 Live Camera Feed")
        video_placeholder = st.empty()

    with col2:
        st.subheader("⚙️ Controls")
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("▶️ Start", use_container_width=True, type="primary", 
                         disabled=st.session_state.run_camera):
                st.session_state.run_camera = True
                st.rerun()
        
        with col_btn2:
            if st.button("⏹️ Stop", use_container_width=True, type="secondary", disabled=not st.session_state.run_camera):
                st.session_state.run_camera = False
                st.session_state.last_faces = []
                if "cap" in st.session_state and st.session_state.cap.isOpened():
                    st.session_state.cap.release()
                st.rerun()
        
        st.divider()
        
        st.subheader("Stats")
        fps_text = st.empty()
        faces_text = st.empty()

    if st.session_state.run_camera:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            st.error("Could not open camera")
            st.session_state.run_camera = False
            st.rerun()
        else:
            frame_count = 0
            fps_list = deque(maxlen=10)
            
            while st.session_state.run_camera:
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    st.warning(" Failed to read frame, retrying...")
                    time.sleep(0.1)
                    continue
                
                processed_frame = process_frame(frame, frame_count, process_interval=10)
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                
                elapsed = time.time() - start_time
                if elapsed > 0:
                    fps = 1.0 / elapsed
                    fps_list.append(fps)
                    avg_fps = np.mean(fps_list)
                    fps_text.metric("FPS", f"{avg_fps:.1f}")
                
                faces_text.metric("Faces", len(st.session_state.last_faces))
                
                frame_count += 1
                
                if frame_count % 10 == 0:
                    if not st.session_state.run_camera:
                        break
                
                time.sleep(0.001)
            
            cap.release()
            cv2.destroyAllWindows()
            
            if not st.session_state.run_camera:
                st.rerun()
    else:
        video_placeholder.info(" Click **'Start'** to Vibe Check!")
        fps_text.empty()
        faces_text.empty()

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>Built with using Streamlit, TensorFlow, and OpenCV</p>
    <p style="font-size: 0.9rem;">VGG16 Model | Real-time Emotion Detection | Computer Vision</p>
</div>
""", unsafe_allow_html=True)