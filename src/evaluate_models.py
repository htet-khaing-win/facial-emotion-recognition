import tensorflow as tf
import numpy as np
import os

DATA_PATH = "data/processed/"

X_test = np.load(os.path.join(DATA_PATH, "X_test.npy"))
y_test = np.load(os.path.join(DATA_PATH, "y_test_cat.npy"))

models_to_test = {
    "baseline": "models/baseline_cnn.h5",
    "augmented": "models/cnn_augmented.h5"
}

for name, path in models_to_test.items():
    model = tf.keras.models.load_model(path)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"{name} model → Loss: {loss:.4f}, Accuracy: {acc:.4f}")
