import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join("data", "fer2013.csv")
OUTPUT_PATH = "data/processed/"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Convert pixel strings → numpy arrays
pixels = df['pixels'].apply(lambda x: np.array(x.split(), dtype='float32'))
images = np.stack(pixels.values)
images = images.reshape(-1, 48, 48, 1) / 255.0   # normalize 0–1

labels = df['emotion'].values

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

# Save arrays
np.save(os.path.join(OUTPUT_PATH, "X_train.npy"), X_train)
np.save(os.path.join(OUTPUT_PATH, "X_test.npy"), X_test)
np.save(os.path.join(OUTPUT_PATH, "y_train.npy"), y_train)
np.save(os.path.join(OUTPUT_PATH, "y_test.npy"), y_test)

print("Preprocessing complete. Saved arrays to", OUTPUT_PATH)

# Visualize random samples
fig, axes = plt.subplots(1, 5, figsize=(12,3))
for i, ax in enumerate(axes):
    idx = np.random.randint(0, len(X_train))
    ax.imshow(X_train[idx].reshape(48,48), cmap='gray')
    ax.set_title(f"Label: {y_train[idx]}")
    ax.axis("off")
plt.show()
