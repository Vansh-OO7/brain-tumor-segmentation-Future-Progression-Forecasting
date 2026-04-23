# src/analyze_uploaded_h5.py

import h5py
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import os

# ==================================================
# CONFIG
# ==================================================
SEG_MODEL_PATH = "models/segmentation_model_30k.h5"
GROWTH_MODEL_PATH = "models/future_growth_model.pkl"

IMG_SIZE = 128
ORIG_SIZE = 240
THRESHOLD = 0.3

OUTPUT_IMAGE = "outputs/Sample_output_1/Sample_segemented_and_prediction.png"

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
# LOAD MODELS
# ==================================================
seg_model = tf.keras.models.load_model(
    SEG_MODEL_PATH,
    custom_objects={
        "dice_coef": dice_coef,
        "bce_dice_loss": bce_dice_loss
    }
)

growth_model = joblib.load(GROWTH_MODEL_PATH)

# ==================================================
# MAIN FUNCTION
# ==================================================
def analyze_uploaded_h5(file_path):
    os.makedirs("outputs", exist_ok=True)

    # ----------------------------------------------
    # LOAD H5
    # ----------------------------------------------
    with h5py.File(file_path, "r") as f:
        img = f["image"][:]

    original_img = img[:, :, 0]

    # ----------------------------------------------
    # PREPROCESS
    # ----------------------------------------------
    img = img[:, :, :4]

    img_resized = tf.image.resize(img, (IMG_SIZE, IMG_SIZE)).numpy()
    img_resized = img_resized.astype(np.float32)

    img_resized = (img_resized - img_resized.min()) / (
        img_resized.max() - img_resized.min() + 1e-8
    )

    # ----------------------------------------------
    # SEGMENTATION
    # ----------------------------------------------
    pred = seg_model.predict(
        np.expand_dims(img_resized, axis=0),
        verbose=0
    )[0]

    pred_mask = (pred > THRESHOLD).astype(np.uint8)

    pred_mask_240 = tf.image.resize(
        pred_mask,
        (ORIG_SIZE, ORIG_SIZE),
        method="nearest"
    ).numpy()

    pred_mask_240 = pred_mask_240[:, :, 0].astype(np.uint8)

    # ----------------------------------------------
    # AREA
    # ----------------------------------------------
    area_pixels = int(np.sum(pred_mask_240))
    area_cm2 = round(area_pixels / 100, 2)

    # ----------------------------------------------
    # FUTURE PREDICTION
    # ----------------------------------------------
    x = np.array([[area_cm2]], dtype=float)
    future = growth_model.predict(x)[0]

    short_term = round(float(future[0]), 2)
    mid_term   = round(float(future[1]), 2)
    long_term  = round(float(future[2]), 2)

    # ----------------------------------------------
    # GROWTH %
    # ----------------------------------------------
    if area_cm2 > 0:
        g_short = round(((short_term - area_cm2) / area_cm2) * 100, 2)
        g_mid   = round(((mid_term - area_cm2) / area_cm2) * 100, 2)
        g_long  = round(((long_term - area_cm2) / area_cm2) * 100, 2)
    else:
        g_short = g_mid = g_long = 0.0

    # ----------------------------------------------
    # PROGRESSION STATUS
    # ----------------------------------------------
    if g_long <= -10:
        status = "Regressive Trend"
    elif g_long <= 5:
        status = "Stable Disease"
    elif g_long <= 20:
        status = "Mild Progressive Trend"
    elif g_long <= 40:
        status = "Moderate Progressive Trend"
    else:
        status = "Rapid Progressive Trend"

    # ----------------------------------------------
    # SAVE VISUAL RESULT
    # ----------------------------------------------
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.imshow(original_img, cmap="gray")
    plt.title("MRI Scan")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(original_img, cmap="gray")
    plt.imshow(pred_mask_240, cmap="jet", alpha=0.5)
    plt.title("Segmented Tumor")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=150, bbox_inches="tight")
    plt.close()

    # ----------------------------------------------
    # RETURN RESULT
    # ----------------------------------------------
    return {
        "segmented_image_path": OUTPUT_IMAGE,

        "current_area_cm2": area_cm2,

        "short_term_cm2": short_term,
        "mid_term_cm2": mid_term,
        "long_term_cm2": long_term,

        "growth_short_term_percent": g_short,
        "growth_mid_term_percent": g_mid,
        "growth_long_term_percent": g_long,

        "progression_status": status
    }

# ==================================================
# DIRECT TEST
# ==================================================
if __name__ == "__main__":
    file_path = input("Enter .h5 file path: ").strip()

    result = analyze_uploaded_h5(file_path)

    print("\n===== FINAL RESULT =====")
    for k, v in result.items():
        print(f"{k}: {v}")