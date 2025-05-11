# # evaluate.py (Single Image Inference)

import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# === Paths ===
image_path = r'C:\Users\Kushal Vishwakarma\Downloads\car_damage\car_damage\test\images\7208.jpg'

# Automatically find model and label encoder in current directory or subfolders
def find_file(filename, search_dir="."):
    for root, _, files in os.walk(search_dir):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"'{filename}' not found in '{search_dir}'.")

# Find model and encoder paths
model_path = find_file("car_damage_model_final.h5")
label_encoder_path = find_file("label_encoder.pkl")

# === Load Label Encoder and Model ===
label_encoder = joblib.load(label_encoder_path)
model = load_model(model_path)

# === Load and Predict ===
img = load_img(image_path, target_size=(224, 224))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)
predicted_index = np.argmax(pred)
predicted_label = label_encoder.inverse_transform([predicted_index])[0]

# === Show Prediction ===
plt.figure(figsize=(6, 6))
plt.imshow(img)
plt.title(f"Predicted: {predicted_label}")
plt.axis('off')
plt.show()

