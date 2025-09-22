import os
import numpy as np
import tensorflow as tf

DATA_PATH = "data/processed/"

X_test = np.load(os.path.join(DATA_PATH, "X_test.npy"))
y_test = np.load(os.path.join(DATA_PATH, "y_test_cat.npy"))

models_to_compare = {
    "baseline": "models/baseline_cnn.h5",
    "augmented_best": "models/cnn_augmented_best.h5",
    "deeper_best": "models/deeper_cnn_best.h5",
    "callbacks_best": "models/cnn_callbacks_best.h5"
}

results = {}

for name, path in models_to_compare.items():
    if os.path.exists(path):
        model = tf.keras.models.load_model(path)
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        results[name] = {"loss": loss, "accuracy": acc}
        print(f"{name}: Loss={loss:.4f}, Accuracy={acc:.4f}")
    else:
        print(f" Skipping {name}, file not found.")

# Save results to file
with open("results/model_comparison.txt", "w") as f:
    for name, metrics in results.items():
        f.write(f"{name}: Loss={metrics['loss']:.4f}, Accuracy={metrics['accuracy']:.4f}\n")