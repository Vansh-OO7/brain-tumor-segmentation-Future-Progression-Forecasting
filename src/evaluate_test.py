# src/evaluate_test.py

import pandas as pd
import numpy as np
import h5py
import tensorflow as tf

# ==================================================
# CONFIG
# ==================================================
TEST_CSV   = "Datasets/test_30k.csv"
MODEL_PATH = "models/segmentation_model_30k.h5"

IMG_SIZE = 128
BATCH_SIZE = 8

# ==================================================
# METRICS / LOSS
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

print("Model loaded successfully.")

# ==================================================
# LOAD TEST FILES
# ==================================================
test_df = pd.read_csv(TEST_CSV)
test_files = test_df["filepath"].tolist()

print("Test samples:", len(test_files))

# ==================================================
# DATA LOADER
# ==================================================
def load_h5(file_path):
    with h5py.File(file_path, "r") as f:
        img = f["image"][:]
        mask = f["mask"][:]

    img = img[:, :, :4]
    mask = np.max(mask, axis=-1, keepdims=True)

    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE)).numpy()
    mask = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE)).numpy()

    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    mask = (mask > 0).astype(np.float32)

    return img, mask

# ==================================================
# GENERATOR
# ==================================================
def data_generator(file_list, batch_size=BATCH_SIZE):
    while True:
        for i in range(0, len(file_list), batch_size):
            batch_files = file_list[i:i+batch_size]

            X_batch = []
            Y_batch = []

            for file in batch_files:
                img, mask = load_h5(file)
                X_batch.append(img)
                Y_batch.append(mask)

            yield np.array(X_batch), np.array(Y_batch)

# ==================================================
# EVALUATE
# ==================================================
test_steps = len(test_files) // BATCH_SIZE

results = model.evaluate(
    data_generator(test_files, BATCH_SIZE),
    steps=test_steps,
    verbose=1
)

print("\n===== FINAL TEST RESULTS =====")
print("Test Loss     :", round(results[0], 4))
print("Test Accuracy :", round(results[1], 4))
print("Test Dice     :", round(results[2], 4))