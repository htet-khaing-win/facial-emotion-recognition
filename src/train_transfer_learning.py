import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import cv2

DATA_PATH = "data/processed/"
X_train = np.load(os.path.join(DATA_PATH, "X_train.npy"))
X_test = np.load(os.path.join(DATA_PATH, "X_test.npy"))
y_train = np.load(os.path.join(DATA_PATH, "y_train_cat.npy"))
y_test = np.load(os.path.join(DATA_PATH, "y_test_cat.npy"))

def resize_images(X, target_size=(96, 96)):
    """Resize images to target size"""
    resized = np.zeros((X.shape[0], target_size[0], target_size[1], X.shape[3]))
    for i in range(X.shape[0]):
        # Squeeze to remove single channel dimension, then resize
        img = np.squeeze(X[i])  # (48,48,1) -> (48,48)
        resized_img = cv2.resize(img, target_size)
        resized[i] = np.expand_dims(resized_img, axis=-1)  # Add channel back
    return resized

print("Resizing images from 48x48 to 96x96...")
X_train = resize_images(X_train, (96, 96))
X_test = resize_images(X_test, (96, 96))
print(f"New shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)
datagen.fit(X_train)

# Expand grayscale to RGB for MobileNetV2 

X_train_rgb = np.repeat(X_train, 3, axis= -1)
X_test_rgb = np.repeat(X_test, 3, axis= -1)

base_model = MobileNetV2(
    weights = "imagenet",
    include_top = False,
    input_shape = (96, 96, 3)  # Changed from (48, 48, 3) to (96, 96, 3)
)
base_model.trainable = True

# Freeze most layers, unfreeze last 30
for layer in base_model.layers[:-30]:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(7, activation="softmax")
])


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

checkpoint_cb = ModelCheckpoint(
    "models/mobilenetv2_finetuned_best.h5",
    save_best_only = True,
    monitor = "val_accuracy",
    mode = "max"
) 

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3)
]

history = model.fit(
    datagen.flow(X_train_rgb, y_train, batch_size=64),
    validation_data=(X_test_rgb, y_test),
    epochs=50,
    callbacks = callbacks
)

import matplotlib.pyplot as plt

# Simple training history plot
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/mobilenetv2_finetuned_final.h5")

# Evaluate
loss, acc = model.evaluate(X_test_rgb, y_test, verbose=0)
print(f" MobileNetV2 (fine-tuned & Resized) Test Accuracy: {acc:.4f}")