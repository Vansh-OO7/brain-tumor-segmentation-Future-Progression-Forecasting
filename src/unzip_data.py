import zipfile
import os

zip_path = "Datasets/Brain_Tumor_Dataset.zip"
extract_to = "Datasets/extracted"

os.makedirs(extract_to, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print("Dataset extracted successfully!")