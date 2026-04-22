# src/train_segmentation.py

import os
import pandas as pd
import numpy as np
import h5py
import tensorflow as tf

# ==================================================
# CONFIG
# ==================================================
TRAIN_CSV = "Datasets/train.csv"
VAL_CSV   = "Datasets/val.csv"

MODEL_PATH = "models/segmentation_model.h5"

IMG_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-3

# ==================================================
# LOAD CSV FILE LISTS
# ==================================================
train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)

train_files = train_df["filepath"].tolist()
val_files   = val_df["filepath"].tolist()

print("Train samples:", len(train_files))
print("Val samples  :", len(val_files))

# ==================================================
# DATA LOADER
# ==================================================
def load_h5(file_path):
    with h5py.File(file_path, "r") as f:
        img = f["image"][:]   # (240,240,4)
        mask = f["mask"][:]  # (240,240,3)

    # Use all 4 channels
    img = img[:, :, :4]

    # Merge mask channels -> binary mask
    mask = np.max(mask, axis=-1, keepdims=True)

    # Resize
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE)).numpy()
    mask = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE)).numpy()

    # Normalize image
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # Binary mask
    mask = (mask > 0).astype(np.float32)

    return img, mask

# ==================================================
# GENERATOR
# ==================================================
def data_generator(file_list, batch_size=BATCH_SIZE):
    while True:
        np.random.shuffle(file_list)

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
# STEPS
# ==================================================
train_steps = len(train_files) // BATCH_SIZE
val_steps   = max(1, len(val_files) // BATCH_SIZE)

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
# LIGHT U-NET
# ==================================================
inputs = tf.keras.layers.Input((IMG_SIZE, IMG_SIZE, 4))

# Encoder
c1 = tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same")(inputs)
c1 = tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same")(c1)
p1 = tf.keras.layers.MaxPooling2D()(c1)

c2 = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(p1)
c2 = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(c2)
p2 = tf.keras.layers.MaxPooling2D()(c2)

# Bottleneck
bn = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(p2)
bn = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(bn)

# Decoder
u1 = tf.keras.layers.UpSampling2D()(bn)
u1 = tf.keras.layers.Concatenate()([u1, c2])
c3 = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(u1)
c3 = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(c3)

u2 = tf.keras.layers.UpSampling2D()(c3)
u2 = tf.keras.layers.Concatenate()([u2, c1])
c4 = tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same")(u2)
c4 = tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same")(c4)

# Output
outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(c4)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=bce_dice_loss,
    metrics=["accuracy", dice_coef]
)

model.summary()

# ==================================================
# CALLBACKS
# ==================================================
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        save_best_only=True,
        monitor="val_dice_coef",
        mode="max",
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_dice_coef",
        patience=3,
        mode="max",
        restore_best_weights=True,
        verbose=1
    )
]

# ==================================================
# TRAIN
# ==================================================
history = model.fit(
    data_generator(train_files, BATCH_SIZE),
    steps_per_epoch=train_steps,
    validation_data=data_generator(val_files, BATCH_SIZE),
    validation_steps=val_steps,
    epochs=EPOCHS,
    callbacks=callbacks
)

print("\nTraining complete.")
print("Best model saved to:", MODEL_PATH)