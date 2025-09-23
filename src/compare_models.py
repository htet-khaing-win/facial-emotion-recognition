import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

DATA_PATH = "data/processed/"
X_test = np.load(os.path.join(DATA_PATH, "X_test.npy"))
y_test = np.load(os.path.join(DATA_PATH, "y_test_cat.npy"))

# Expand grayscale → RGB for MobileNetV2 models
X_test_rgb = np.repeat(X_test, 3, axis=-1)

models_to_compare = {
    "cnn_baseline": ("models/baseline_cnn.h5", X_test),
    "cnn_augmented": ("models/cnn_augmented_best.h5", X_test),
    "cnn_deeper": ("models/deeper_cnn_best.h5", X_test),
    "cnn_callbacks": ("models/cnn_callbacks_best.h5", X_test),
    "mobilenet_frozen": ("models/mobilenetv2_frozen.h5", X_test_rgb),
    "mobilenet_finetuned": ("models/mobilenetv2_finetuned.h5", X_test_rgb),
}

results = {}

for name, (path, data) in models_to_compare.items():
    if os.path.exists(path):
        model = tf.keras.models.load_model(path)
        loss, acc = model.evaluate(data, y_test, verbose=0)
        results[name] = {"loss": loss, "accuracy": acc}
        print(f"{name}: Loss={loss:.4f}, Accuracy={acc:.4f}")
    else:
        print(f" Skipping {name}, file not found.")

# Save results
os.makedirs("results", exist_ok=True)
with open("results/final_comparison.txt", "w") as f:
    for name, metrics in results.items():
        f.write(f"{name}: Loss={metrics['loss']:.4f}, Accuracy={metrics['accuracy']:.4f}\n")

# Bar chart visualization
names = list(results.keys())
accuracies = [results[n]["accuracy"] for n in names]

plt.figure(figsize=(10,6))
plt.bar(names, accuracies)
plt.xticks(rotation=45)
plt.ylabel("Test Accuracy")
plt.title("CNNs vs Transfer Learning (FER)")
plt.tight_layout()
plt.savefig("results/final_comparison.png")
print(" Comparison complete. Results saved in results/")
