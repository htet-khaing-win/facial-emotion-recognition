import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import gc

# Load test data
DATA_PATH = "data/processed/"
X_test = np.load(os.path.join(DATA_PATH, "X_test.npy"))   
y_test = np.load(os.path.join(DATA_PATH, "y_test_cat.npy"))

print("Loading and evaluating models...")

# Define models
models_to_compare = {
    "CNN (Fine Tuned)": {"path": "models/cnn_callbacks_best.h5", "size": 48, "grayscale": True},
    "MobileNetV2 (Fine Tuned)": {"path": "models/mobilenetv2_finetuned_224x224_best.h5", "size": 224, "grayscale": False},
    "VGG16 (Fine Tuned)": {"path": "models/vgg16_conservative_img160_dr0.28_lr0.0001_best.h5", "size": 160, "grayscale": False},
}

def resize_images(X, target_size, is_grayscale=False):
    """Resize images efficiently"""
    if is_grayscale:
        # Keep as grayscale (48x48x1)
        resized = np.zeros((X.shape[0], target_size, target_size, 1), dtype=np.float32)
        for i in range(X.shape[0]):
            img = np.squeeze(X[i])
            resized_img = cv2.resize(img, (target_size, target_size))
            resized[i] = resized_img[:, :, np.newaxis]
    else:
        # Convert to RGB
        resized = np.zeros((X.shape[0], target_size, target_size, 3), dtype=np.float32)
        for i in range(X.shape[0]):
            img = np.squeeze(X[i])
            resized_img = cv2.resize(img, (target_size, target_size))
            resized[i] = np.stack([resized_img, resized_img, resized_img], axis=-1)
    return resized

results = {}

# Evaluate each model
for name, config in models_to_compare.items():
    path = config["path"]
    size = config["size"]
    
    if not os.path.exists(path):
        print(f"Skipping {name} - file not found")
        continue
    
    try:
        print(f"\nEvaluating {name}...")
        
        # Prepare data
        is_grayscale = config.get("grayscale", False)
        if is_grayscale and size == 48:
            data = X_test  # Keep as 48x48x1
        else:
            data = resize_images(X_test, size, is_grayscale=False)
        
        # Load and evaluate
        model = tf.keras.models.load_model(path, compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        loss, acc = model.evaluate(data, y_test, batch_size=32, verbose=0)
        results[name] = {"loss": loss, "accuracy": acc}
        print(f"  Accuracy: {acc*100:.2f}%, Loss: {loss:.4f}")
        
        # Clean up memory
        del model, data
        gc.collect()
        tf.keras.backend.clear_session()
        
    except Exception as e:
        print(f"  Failed: {str(e)[:100]}")

# Exit if no models loaded
if not results:
    print("\nNo models successfully evaluated")
    exit()

# Prepare data for plotting
sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
names = [name for name, _ in sorted_results]
accuracies = [m['accuracy'] * 100 for _, m in sorted_results]

print(f"\nModels evaluated: {len(results)}")
for name, acc in zip(names, accuracies):
    print(f"  {name}: {acc:.2f}%")

plt.xkcd()  

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(names))
bars = ax.bar(x, accuracies, color='steelblue', edgecolor='black', width=0.6, linewidth=2)

# Highlight best with consistent edge color
bars[0].set_color('seagreen')
bars[0].set_edgecolor('black')
bars[0].set_linewidth(2)

# Zoom in to show differences clearly
min_acc = min(accuracies)
max_acc = max(accuracies)
margin = (max_acc - min_acc) * 0.2  # Add 20% margin
ax.set_ylim(min_acc - margin, max_acc + margin)

# Labels and formatting
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Comparison between the models', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=20, ha='right')
# ax.grid(axis='y', alpha=0.3)

# Add accuracy values on bars
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + margin*0.15,
            f'{acc:.2f}%', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()

# Save
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/model_comparison.png", dpi=300, bbox_inches='tight')
print("\nChart saved to figures/model_comparison.png")
plt.show()