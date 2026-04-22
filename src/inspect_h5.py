import os
import h5py

folder = "Datasets/extracted/BraTS2020_training_data/content/data"

files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".h5")]

print("Total Files:", len(files))

sample = files[0]
print("Sample:", sample)

with h5py.File(sample, "r") as f:
    print("Keys:", list(f.keys()))
    for key in f.keys():
        print(key, f[key].shape, f[key].dtype)