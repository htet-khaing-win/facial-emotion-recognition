import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

DATA_PATH = "../data/processed/"

#loading data

X_train = np.load(os.path.join(DATA_PATH, "X_train.npy"))
X_test  = np.load(os.path.join(DATA_PATH,"X_test.npy"))
y_train = np.load(os.path.join(DATA_PATH, "y_train.npy"))
y_test  = np.load(os.path.join(DATA_PATH,"y_test.npy"))

print("Train shape: ", X_train.shape, y_train.shape)


# Baseline CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(7, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# # Train
# history = model.fit(
#     X_train, y_train,
#     validation_data=(X_test, y_test),
#     epochs=5,
#     batch_size=64
# )

# After Training

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10, #increase to see clearer trends
    batch_size=64
)

# Plot accuracy and loss

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label= 'train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend(); plt.title("Accuracy")

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label = 'train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend(); plt.title("Loss")
plt.show()

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/baseline_cnn.h5")

print("Model trained and saved at models/baseline_cnn")