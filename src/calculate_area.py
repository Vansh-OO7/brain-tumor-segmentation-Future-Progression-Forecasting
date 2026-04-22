# src/calculate_tumor_area.py

import pandas as pd
import numpy as np
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt

# ==================================================
# CONFIG
# ==================================================
TEST_CSV   = "Datasets/test_30k.csv"
MODEL_PATH = "models/segmentation_model_30k.h5"

IMG_SIZE = 128
THRESHOLD = 0.2
SAMPLE_INDEX = 50  # change to test another sample

# ==================================================
# CUSTOM OBJECTS
# ==================================================
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred)

    return (2. * intersection + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    )

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

# ==================================================
# LOAD MODEL
# ==================================================
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "dice_coef": dice_coef,
        "bce_dice_loss": bce_dice_loss
    }
)

print("Model loaded.")

# ==================================================
# LOAD TEST SAMPLE
# ==================================================
df = pd.read_csv(TEST_CSV)

tumor_candidates = []

for i in range(len(df)):
    file_path = df.iloc[i]["filepath"]

    with h5py.File(file_path, "r") as f:
        mask = f["mask"][:]

    if np.sum(mask) > 0:
        tumor_candidates.append(file_path)

# choose random tumor sample every run
file_path = np.random.choice(tumor_candidates)

print("Random tumor sample selected:")

print("Using file:", file_path)

with h5py.File(file_path, "r") as f:
    img = f["image"][:]
    true_mask = f["mask"][:]

# ==================================================
# PREPROCESS
# ==================================================
img = img[:, :, :4]
true_mask = np.max(true_mask, axis=-1, keepdims=True)

img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE)).numpy()
true_mask = tf.image.resize(true_mask, (IMG_SIZE, IMG_SIZE)).numpy()

img = img.astype(np.float32)
img = (img - img.min()) / (img.max() - img.min() + 1e-8)
true_mask = (true_mask > 0).astype(np.uint8)

# ==================================================
# PREDICT MASK
# ==================================================
pred = model.predict(np.expand_dims(img, axis=0), verbose=0)[0]
pred_mask = (pred > THRESHOLD).astype(np.uint8)

# ==================================================
# AREA CALCULATION
# ==================================================
tumor_pixels = int(np.sum(pred_mask))
total_pixels = IMG_SIZE * IMG_SIZE
tumor_percent = (tumor_pixels / total_pixels) * 100

true_pixels = int(np.sum(true_mask))

print("\n===== TUMOR AREA RESULT =====")
print("Predicted Tumor Pixels :", tumor_pixels)
print("True Tumor Pixels      :", true_pixels)
print("Tumor Area Percentage  :", round(tumor_percent, 2), "%")

# ==================================================
# VISUALIZATION
# ==================================================
plt.figure(figsize=(14,4))

plt.subplot(1,3,1)
plt.imshow(img[:, :, 0], cmap="gray")
plt.title("MRI Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(true_mask[:, :, 0], cmap="gray")
plt.title("True Mask")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(pred_mask[:, :, 0], cmap="gray")
plt.title("Predicted Mask")
plt.axis("off")

plt.tight_layout()
plt.show()