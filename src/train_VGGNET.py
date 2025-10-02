import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import cv2
import gc

# GPU Configuration
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         print(f"GPU configured: {gpus}")
#     except RuntimeError as e:
#         print(e)


CONFIG = {
    'IMG_SIZE': 160,
    'BATCH_SIZE': 32,
    'EPOCHS': 100,
    'LEARNING_RATE': 0.0001,  
    'DROPOUT_RATE': 0.28,  
    'DENSE_UNITS': 512,
    'FREEZE_LAYERS': False,
    'UNFREEZE_LAST_N': 6,
    'L2_REG': 0.000075,
}

print("="*60)
print("VGG16 TRAINING")
print("="*60)
print("Configuration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")
print("="*60)

DATA_PATH = "data/processed/"

# Load data
print("\nLoading data...")
X_train = np.load(os.path.join(DATA_PATH, "X_train.npy"))
X_test = np.load(os.path.join(DATA_PATH, "X_test.npy"))
y_train = np.load(os.path.join(DATA_PATH, "y_train_cat.npy"))
y_test = np.load(os.path.join(DATA_PATH, "y_test_cat.npy"))

print(f"Loaded: {X_train.shape}, {X_test.shape}")

# Resize images
def resize_images(X, target_size):
    resized = np.zeros((X.shape[0], target_size, target_size, 3), dtype=np.float32)
    for i in range(X.shape[0]):
        img = np.squeeze(X[i])
        resized_img = cv2.resize(img, (target_size, target_size))
        resized[i] = np.stack([resized_img, resized_img, resized_img], axis=-1)
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1}/{X.shape[0]}")
    return resized

print(f"\nResizing to {CONFIG['IMG_SIZE']}x{CONFIG['IMG_SIZE']}...")
X_train_rgb = resize_images(X_train, CONFIG['IMG_SIZE'])
X_test_rgb = resize_images(X_test, CONFIG['IMG_SIZE'])

del X_train, X_test
gc.collect()

print(f"RGB shapes: {X_train_rgb.shape}, {X_test_rgb.shape}")

# Data Augmentation 
print("\nSetting up data augmentation...")
datagen = ImageDataGenerator(
    rotation_range=20, 
    width_shift_range=0.2,  
    height_shift_range=0.2,  
    zoom_range=0.2,  
    horizontal_flip=True
)
datagen.fit(X_train_rgb)

# Build VGG16
print("\nBuilding VGG16 model...")
base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'], 3)
)

# Unfreeze 
base_model.trainable = True
trainable_count = 0
for layer in base_model.layers[:-CONFIG['UNFREEZE_LAST_N']]:
    layer.trainable = False
for layer in base_model.layers[-CONFIG['UNFREEZE_LAST_N']:]:
    layer.trainable = True
    trainable_count += 1

print(f"Total VGG16 layers: {len(base_model.layers)}")
print(f"Trainable layers: {trainable_count}")

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(CONFIG['DENSE_UNITS'], activation='relu'),  
    layers.Dropout(CONFIG['DROPOUT_RATE']),
    layers.Dense(7, activation='softmax')
])

print("\nModel Architecture:")
model.summary()

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG['LEARNING_RATE']),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

model_name = f"vgg16_conservative_img{CONFIG['IMG_SIZE']}_dr{CONFIG['DROPOUT_RATE']}_lr{CONFIG['LEARNING_RATE']}"

callbacks = [
    ModelCheckpoint(
        f"models/{model_name}_best.h5",
        save_best_only=True,
        monitor="val_accuracy",
        mode="max",
        verbose=1
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=5,  
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=5,
        min_lr=1e-8,
        verbose=1
    )
]

# Train
print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)

history = model.fit(
    datagen.flow(X_train_rgb, y_train, batch_size=CONFIG['BATCH_SIZE']),
    validation_data=(X_test_rgb, y_test),
    validation_batch_size=CONFIG['BATCH_SIZE'],
    epochs=CONFIG['EPOCHS'],
    callbacks=callbacks,
    steps_per_epoch=len(X_train_rgb) // CONFIG['BATCH_SIZE'],
    verbose=1
)

print("\n" + "="*60)
print("TRAINING COMPLETE - EVALUATING")
print("="*60)

# Clear memory
del X_train_rgb
gc.collect()
tf.keras.backend.clear_session()

# Load best model
print("Reloading best model...")
best_model = tf.keras.models.load_model(f"models/{model_name}_best.h5")

# Evaluate
print("Evaluating on test set...")
loss, acc = best_model.evaluate(
    X_test_rgb, 
    y_test, 
    batch_size=CONFIG['BATCH_SIZE'],
    verbose=1
)

# Results
print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print(f"Test Loss: {loss:.4f}")
best_val_acc = max(history.history['val_accuracy'])
print(f"Best Val Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")

print(f" Achieved: {best_val_acc*100:.2f}%")
if best_val_acc >= 0.70:
    print(" TARGET REACHED!")
else:
    print(f" Gap: {(0.70 - best_val_acc)*100:.2f}%")
print("="*60)

# Overfitting analysis
# train_acc = history.history['accuracy'][-5:]
# val_acc = history.history['val_accuracy'][-5:]
# avg_train = np.mean(train_acc)
# avg_val = np.mean(val_acc)
# gap = (avg_train - avg_val) * 100

# print(f"\nOverfitting Analysis (Last 5 epochs):")
# print(f"  Avg Train Acc: {avg_train*100:.2f}%")
# print(f"  Avg Val Acc: {avg_val*100:.2f}%")
# print(f"  Gap: {gap:.2f}%")
# if gap > 8:
#     print("   Overfitting detected")
# elif gap < 2:
#     print("   Underfitting detected")
# else:
#     print("   Good balance!")

# Save results
results = {
    'config': CONFIG,
    'test_accuracy': float(acc),
    'test_loss': float(loss),
    'best_val_accuracy': float(best_val_acc),
    'train_val_gap': float(gap),
}

import json
with open(f'results/{model_name}_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to results/{model_name}_results.json")

# Plot
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Val', linewidth=2)
plt.title(f'VGG16 Accuracy (Best: {best_val_acc:.3f})')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train', linewidth=2)
plt.plot(history.history['val_loss'], label='Val', linewidth=2)
plt.title('VGG16 Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'results/{model_name}_history.png', dpi=200)
print(f"Plot saved to results/{model_name}_history.png")

# Save final model
best_model.save(f"models/{model_name}_final.h5")
print(f"Model saved to models/{model_name}_final.h5")

# print("\n" + "="*60)
# print("EXPERIMENT COMPLETE")
# print("="*60)
# print("\nProgression:")
# print(f"  Frozen VGG16:           45.19%")
# print(f"  This run (conservative): {acc*100:.2f}%")
# print("="*60)
