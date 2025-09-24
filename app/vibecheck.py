import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("models/mobilenetv2_finetuned_224x224_best.h5")

# FER-2013 labels
class_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

st.title("Vibe Check")
st.write("Real-time FER using MobileNetV2 + OpenCV + Streamlit")

option = st.sidebar.selectbox("Choose Mode", ("Webcam", "Upload Image"))

# Helper function
def predict_emotion(face):
    face = cv2.resize(face, (224, 224))
    face = face.astype("float32") / 255.0
    face_rgb = np.repeat(face[..., np.newaxis], 3, -1)  # grayscale → RGB
    face_rgb = np.expand_dims(face_rgb, axis=0)
    preds = model.predict(face_rgb, verbose=0)[0]
    label = class_labels[np.argmax(preds)]
    confidence = np.max(preds)
    return label, confidence

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload a face image", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x,y,w,h) in faces:
            face = gray[y:y+h, x:x+w]
            label, conf = predict_emotion(face)
            cv2.rectangle(img_array, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(img_array, f"{label} ({conf:.2f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        st.image(img_array, caption="Predicted Emotion", use_column_width=True)

elif option == "Webcam":
    st.write("Press **Start** to Vibe Check!")
    run = st.checkbox("Start")
    if run:
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        frame_placeholder = st.empty()
        while run:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            for (x,y,w,h) in faces:
                face = gray[y:y+h, x:x+w]
                label, conf = predict_emotion(face)
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(frame, f"{label} ({conf:.2f})", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        cap.release()
