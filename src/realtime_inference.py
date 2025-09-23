import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import time

# Load best model
print("Loading model...")
model = tf.keras.models.load_model("models/mobilenetv2_finetuned_224x224_best.h5")

# Emotion Labels
class_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Simple optimization settings
PROCESS_EVERY_N_FRAMES = 3  # Only process every 3rd frame
PREDICTION_BUFFER_SIZE = 3  # Keep last 3 predictions for stability

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce camera buffer

# Tracking variables
frame_count = 0
last_predictions = deque(maxlen=PREDICTION_BUFFER_SIZE)
last_faces = []
fps_times = deque(maxlen=30)

print("Starting optimized inference... Press 'q' to quit")

while True:
    start_time = time.time()
    
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_predictions = []
    
    # Only run face detection and prediction every N frames
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Face detection with relaxed parameters
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=4,
            minSize=(50, 50)
        )
        
        for (x, y, w, h) in faces:
            # Extract and process ROI
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (224, 224))
            roi_gray = roi_gray.astype("float32") / 255.0
            roi_rgb = np.repeat(roi_gray[..., np.newaxis], 3, -1)
            roi_rgb = np.expand_dims(roi_rgb, axis=0)

            # Get prediction
            preds = model.predict(roi_rgb, verbose=0)[0]
            label = class_labels[np.argmax(preds)]
            confidence = np.max(preds)
            
            current_predictions.append({
                'bbox': (x, y, w, h),
                'label': label,
                'confidence': confidence
            })
        
        # Update prediction buffer
        if current_predictions:
            last_predictions.append(current_predictions)
        
        last_faces = faces
    
    # Draw results using most recent predictions
    if last_predictions and len(last_predictions) > 0:
        # Get most stable prediction (majority vote from recent frames)
        recent_preds = list(last_predictions)
        
        if len(recent_preds) >= 2:
            # Use second most recent for stability (avoid flickering)
            stable_preds = recent_preds[-2]
        else:
            stable_preds = recent_preds[-1]
            
        for pred in stable_preds:
            x, y, w, h = pred['bbox']
            label = pred['label']
            confidence = pred['confidence']
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Calculate FPS
    end_time = time.time()
    frame_time = end_time - start_time
    if frame_time > 0:
        fps = 1.0 / frame_time
        fps_times.append(fps)
        avg_fps = sum(fps_times) / len(fps_times)
        
        # Display FPS
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Vibe Check", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()