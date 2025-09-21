import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

DATA_PATH = "data/processed/"
MODEL_PATH = "models/cnn_augmented_best.h5"

# LOAD data

X_test = np.load(os.path.join(DATA_PATH, "X_test.npy"))
y_test = np.load(os.path.join(DATA_PATH, "y_test.npy"))
y_test_cat = np.load(os.path.join(DATA_PATH, "y_test_cat.npy"))

# load model
model = tf.keras.models.load_model(MODEL_PATH)

# Predictions

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis= 1)

# Confusion Matrix

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot= True, fmt="d", cmap= "Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Classification Report

print("Classification Report: ")
print(classification_report(y_test,y_pred,digits=4))