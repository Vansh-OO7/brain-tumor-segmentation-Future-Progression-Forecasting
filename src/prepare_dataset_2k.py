# src/prepare_dataset.py

import os
import random
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ==================================================
# CONFIG
# ==================================================
DATA_PATH = "Datasets/extracted/BraTS2020_training_data/content/data"
OUTPUT_DIR = "Datasets"

TOTAL_SAMPLES = 2000        # total subset size
MIN_TUMOR = 600            # guarantee enough tumor slices
RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ==================================================
# GET FILES
# ==================================================
files = [
    os.path.join(DATA_PATH, f)
    for f in os.listdir(DATA_PATH)
    if f.endswith(".h5")
]

random.shuffle(files)

print("Total files available:", len(files))

# ==================================================
# COLLECT DATASET (MEDICAL-AWARE SAMPLING)
# ==================================================
tumor_files = []
non_tumor_files = []

checked = 0

for file in files:
    with h5py.File(file, "r") as f:
        mask = f["mask"][:]

    has_tumor = np.sum(mask) > 0

    if has_tumor:
        tumor_files.append(file)
    else:
        non_tumor_files.append(file)

    checked += 1

    if checked % 500 == 0:
        print(
            f"Checked {checked} | "
            f"Tumor={len(tumor_files)} | "
            f"NonTumor={len(non_tumor_files)}"
        )

    # Stop when enough tumor found and enough total found
    if len(tumor_files) >= MIN_TUMOR and \
       (len(tumor_files) + len(non_tumor_files)) >= TOTAL_SAMPLES:
        print("Required subset collected. Early stopping.")
        break

# ==================================================
# BUILD FINAL SUBSET
# ==================================================
selected = []

# Add guaranteed tumor samples
selected.extend(tumor_files[:MIN_TUMOR])

remaining_slots = TOTAL_SAMPLES - len(selected)

# Remaining pool = leftover tumor + all non-tumor collected
remaining_pool = tumor_files[MIN_TUMOR:] + non_tumor_files
random.shuffle(remaining_pool)

selected.extend(remaining_pool[:remaining_slots])

# Safety
selected = selected[:TOTAL_SAMPLES]
random.shuffle(selected)

# Labels for stratified split
labels = []

for file in selected:
    with h5py.File(file, "r") as f:
        mask = f["mask"][:]

    labels.append(1 if np.sum(mask) > 0 else 0)

labels = np.array(labels)

print("\nFinal subset size:", len(selected))
print("Tumor samples    :", int(np.sum(labels == 1)))
print("Non-tumor samples:", int(np.sum(labels == 0)))

# ==================================================
# TRAIN / TEMP SPLIT
# ==================================================
train_files, temp_files, train_labels, temp_labels = train_test_split(
    selected,
    labels,
    test_size=(1 - TRAIN_RATIO),
    random_state=RANDOM_SEED,
    stratify=labels
)

# ==================================================
# VAL / TEST SPLIT
# ==================================================
val_portion = VAL_RATIO / (VAL_RATIO + TEST_RATIO)

val_files, test_files, val_labels, test_labels = train_test_split(
    temp_files,
    temp_labels,
    test_size=(1 - val_portion),
    random_state=RANDOM_SEED,
    stratify=temp_labels
)

# ==================================================
# SAVE CSV FILES
# ==================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

pd.DataFrame({
    "filepath": train_files,
    "label": train_labels
}).to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)

pd.DataFrame({
    "filepath": val_files,
    "label": val_labels
}).to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)

pd.DataFrame({
    "filepath": test_files,
    "label": test_labels
}).to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

# ==================================================
# SUMMARY
# ==================================================
print("\nSaved files:")
print("train.csv =", len(train_files))
print("val.csv   =", len(val_files))
print("test.csv  =", len(test_files))

print("\nTrain tumor:", int(np.sum(train_labels == 1)),
      "| non-tumor:", int(np.sum(train_labels == 0)))

print("Val tumor  :", int(np.sum(val_labels == 1)),
      "| non-tumor:", int(np.sum(val_labels == 0)))

print("Test tumor :", int(np.sum(test_labels == 1)),
      "| non-tumor:", int(np.sum(test_labels == 0)))