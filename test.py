# test.py

import os
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# === Paths ===
test_images_dir = r'C:\Users\Kushal Vishwakarma\Downloads\car_damage\car_damage\test\images'
test_csv_path = r'C:\Users\Kushal Vishwakarma\Downloads\car_damage\car_damage\test\test.csv'

# === Auto Path Finder ===
def find_file(filename, search_dir="."):
    for root, _, files in os.walk(search_dir):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"'{filename}' not found in '{search_dir}'.")

# === Locate Model and Label Encoder ===
model_path = find_file("car_damage_model_final.h5")
label_encoder_path = find_file("label_encoder.pkl")

# === Load test CSV ===
df_test = pd.read_csv(test_csv_path)

# === Load and preprocess test images ===
X_test = []
test_filenames = []

for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Loading Test Images"):
    img_path = os.path.join(test_images_dir, row['filename'])
    try:
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        X_test.append(img_array)
        test_filenames.append(row['filename'])
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")

X_test = np.array(X_test)

# === Load model and label encoder ===
model = load_model(model_path)
label_encoder = joblib.load(label_encoder_path)

# === Make predictions ===
predictions = model.predict(X_test)
predicted_indices = np.argmax(predictions, axis=1)
predicted_labels = label_encoder.inverse_transform(predicted_indices)

# === Print predictions (filename + class name) ===
for fname, label in zip(test_filenames, predicted_labels):
    print(f"{fname} → {label}")

