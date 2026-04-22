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
ORIG_SIZE = 240
THRESHOLD = 0.3

# BraTS assumed spacing: 1mm x 1mm
MM2_PER_PIXEL = 1.0

# ==================================================
# CUSTOM OBJECTS
# ==================================================
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2.0 * intersection + smooth) / (
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
# PICK RANDOM TUMOR SAMPLE
# ==================================================
df = pd.read_csv(TEST_CSV)

tumor_candidates = []

for i in range(len(df)):
    file_path = df.iloc[i]["filepath"]

    with h5py.File(file_path, "r") as f:
        mask = f["mask"][:]

    if np.sum(mask) > 0:
        tumor_candidates.append(file_path)

file_path = np.random.choice(tumor_candidates)
print("Using file:", file_path)

# ==================================================
# LOAD DATA
# ==================================================
with h5py.File(file_path, "r") as f:
    img = f["image"][:]       # (240,240,4)
    true_mask = f["mask"][:] # (240,240,3)

# Keep original GT at 240x240
true_mask_240 = (np.max(true_mask, axis=-1) > 0).astype(np.uint8)

# Image preprocess for model
img = img[:, :, :4]
img_resized = tf.image.resize(img, (IMG_SIZE, IMG_SIZE)).numpy()

img_resized = img_resized.astype(np.float32)
img_resized = (img_resized - img_resized.min()) / (
    img_resized.max() - img_resized.min() + 1e-8
)

# ==================================================
# PREDICT MASK
# ==================================================
pred = model.predict(np.expand_dims(img_resized, axis=0), verbose=0)[0]

# Binary mask at 128x128
pred_mask_128 = (pred > THRESHOLD).astype(np.uint8)

# Resize prediction back to original 240x240
pred_mask_240 = tf.image.resize(
    pred_mask_128,
    (ORIG_SIZE, ORIG_SIZE),
    method="nearest"
).numpy()

pred_mask_240 = pred_mask_240[:, :, 0].astype(np.uint8)

# ==================================================
# AREA CALCULATION
# ==================================================
pred_pixels = int(np.sum(pred_mask_240))
true_pixels = int(np.sum(true_mask_240))

pixel_error = pred_pixels - true_pixels
abs_error = abs(pixel_error)

pred_mm2 = pred_pixels * MM2_PER_PIXEL
true_mm2 = true_pixels * MM2_PER_PIXEL

pred_cm2 = pred_mm2 / 100
true_cm2 = true_mm2 / 100

print("\n===== PREDICTED TUMOR AREA =====")
print("Predicted Pixels   :", pred_pixels)
print("Predicted Area (mm²):", round(pred_mm2, 2))
print("Predicted Area (cm²):", round(pred_cm2, 2))

# ==================================================
# VISUALIZATION
# ==================================================
plt.figure(figsize=(15,4))

plt.subplot(1,3,1)
plt.imshow(img[:, :, 0], cmap="gray")
plt.title("MRI Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(true_mask_240, cmap="gray")
plt.title("Ground Truth Mask")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(pred_mask_240, cmap="gray")
plt.title("Predicted Mask")
plt.axis("off")

plt.tight_layout()
plt.show()