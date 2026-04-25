# src/train_segmentation.py

import os
import pandas as pd
import numpy as np
import h5py
import tensorflow as tf

# ==================================================
# CONFIG
# ==================================================
TRAIN_CSV = "Datasets/train_30k.csv"
VAL_CSV   = "Datasets/val_30k.csv"

MODEL_PATH = "models/segmentation_model_30k.h5"

IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 12
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

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
# BLOCK
# ==================================================
def conv_block(x, filters, dropout_rate=0.0):
    x = tf.keras.layers.Conv2D(
        filters, 3, padding="same",
        kernel_initializer="he_normal"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(
        filters, 3, padding="same",
        kernel_initializer="he_normal"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    if dropout_rate > 0:
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    return x

# ==================================================
# U-NET
# ==================================================
inputs = tf.keras.layers.Input((IMG_SIZE, IMG_SIZE, 4))

# Encoder
c1 = conv_block(inputs, 16, dropout_rate=0.05)
p1 = tf.keras.layers.MaxPooling2D()(c1)

c2 = conv_block(p1, 32, dropout_rate=0.10)
p2 = tf.keras.layers.MaxPooling2D()(c2)

# Bottleneck
bn = conv_block(p2, 64, dropout_rate=0.20)

# Decoder
u1 = tf.keras.layers.UpSampling2D()(bn)
u1 = tf.keras.layers.Concatenate()([u1, c2])
c3 = conv_block(u1, 32, dropout_rate=0.10)

u2 = tf.keras.layers.UpSampling2D()(c3)
u2 = tf.keras.layers.Concatenate()([u2, c1])
c4 = conv_block(u2, 16, dropout_rate=0.05)

outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(c4)

model = tf.keras.Model(inputs, outputs)

# ==================================================
# COMPILE
# ==================================================
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

model.compile(
    optimizer=optimizer,
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
        patience=4,
        mode="max",
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_dice_coef",
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
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