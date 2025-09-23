import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import smart_resize


DATA_PATH = "data/processed/"
X_test = np.load(os.path.join(DATA_PATH, "X_test.npy"))         # (N, 48, 48, 1)
y_test = np.load(os.path.join(DATA_PATH, "y_test_cat.npy"))

# Expand grayscale → RGB for transfer learning models
X_test_rgb = np.repeat(X_test, 3, axis=-1)  # (N, 48, 48, 3)


# Models to compare
models_to_compare = {
    "cnn_baseline": "models/baseline_cnn.h5",
    "cnn_augmented": "models/cnn_augmented_best.h5",
    "cnn_deeper": "models/deeper_cnn_best.h5",
    "cnn_callbacks": "models/cnn_callbacks_best.h5",
    "mobilenet_frozen": "models/mobilenetv2_frozen.h5",
    "mobilenet_finetuned": "models/mobilenetv2_finetuned.h5",
    "mobilenet_finetuned 96x96": "models/mobilenetv2_finetuned_final.h5",
    "mobilenet_finetuned 224x224": "models/mobilenetv2_finetuned_224x224_best.h5",
}

results = {}

# Evaluate models
for name, path in models_to_compare.items():
    if os.path.exists(path):
        model = tf.keras.models.load_model(path)
        input_shape = model.input_shape[1:]  # (H, W, C)

        # Decide which test set to feed
        if input_shape == (48, 48, 1):        # CNN grayscale
            data = X_test
        elif input_shape == (48, 48, 3):      # CNN trained on RGB-expanded grayscale
            data = X_test_rgb
        elif input_shape == (96, 96, 3):      # MobileNetV2 small input
            data = smart_resize(X_test_rgb, (96, 96))
        elif input_shape == (224, 224, 3):    # MobileNetV2 large input
            data = smart_resize(X_test_rgb, (224, 224))
        else:
            raise ValueError(f" Unexpected input shape {input_shape} for model {name}")

        # Evaluate
        loss, acc = model.evaluate(data, y_test, verbose=0)
        results[name] = {"loss": loss, "accuracy": acc}
        print(f"{name}: Loss={loss:.4f}, Accuracy={acc:.4f}")
    else:
        print(f" Skipping {name}, file not found.")


# Save results to file
os.makedirs("results", exist_ok=True)
with open("results/final_comparison.txt", "w") as f:
    for name, metrics in results.items():
        f.write(f"{name}: Loss={metrics['loss']:.4f}, Accuracy={metrics['accuracy']:.4f}\n")

# Plot comparison chart
names = list(results.keys())
accuracies = [results[n]["accuracy"] for n in names]

plt.figure(figsize=(10, 6))
plt.bar(names, accuracies)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Test Accuracy")
plt.title("CNNs vs Transfer Learning (FER2013)")
plt.tight_layout()
plt.savefig("results/comparison_v2.png")
print(" Comparison complete. Results saved in results/")
