import cv2
import numpy as np
import tensorflow as tf

# Load best model
model = tf.keras.models.load_model("models/mobilenetv2_finetuned_224x224_best.h5")

# Emotion Labels
class_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0) #0 = default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors= 5, minSize=(48,48))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (224,224))
        roi_gray = roi_gray.astype("float32") / 255.0
        roi_rgb = np.repeat(roi_gray[..., np.newaxis], 3, -1)  # grayscale → RGB
        roi_rgb = np.expand_dims(roi_rgb, axis=0)

        preds = model.predict(roi_rgb, verbose=0)[0]
        label = class_labels[np.argmax(preds)]
        confidence = np.max(preds)

        # Draw bounding box + label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow("Facial Emotion Recognition (Real-Time)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()