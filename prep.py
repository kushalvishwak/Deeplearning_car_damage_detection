# prep.py

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Label mapping dictionary
label_to_cls = {
    1: "crack",
    2: "scratch",
    3: "tire flat",
    4: "dent",
    5: "glass shatter",
    6: "lamp broken"
}

def prepare_data(csv_path, images_dir, img_size=(224, 224)):
    """
    Loads image data and labels, preprocesses them, and splits into training and validation sets.

    Args:
        csv_path (str): Path to the CSV file containing image filenames and labels.
        images_dir (str): Directory containing the image files.
        img_size (tuple): Desired image size (default is 224x224).

    Returns:
        X_train, X_val, y_train, y_val, label_encoder: Numpy arrays and encoder.
    """
    df = pd.read_csv(csv_path)

    # Check for unknown labels
    unique_labels = df['label'].unique()
    unknown_labels = [label for label in unique_labels if label not in label_to_cls]
    if unknown_labels:
        print(f"Warning: Unknown labels found - {unknown_labels}")

    print("Mapped Class Labels:", [label_to_cls.get(label, "Unknown") for label in unique_labels])

    X = []
    y = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading Images"):
        img_path = os.path.join(images_dir, row['filename'])
        try:
            img = load_img(img_path, target_size=img_size)
            img_array = img_to_array(img) / 255.0  # Normalize
            X.append(img_array)
            y.append(row['label'])
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

    X = np.array(X, dtype="float32")
    y = np.array(y)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Filter classes with fewer than 2 samples
    label_counts = Counter(y_encoded)
    valid_mask = np.array([label_counts[label] >= 2 for label in y_encoded])

    X_filtered = X[valid_mask]
    y_filtered = y_encoded[valid_mask]

    # Final train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered
    )

    print("Training set shape:", X_train.shape, y_train.shape)
    print("Validation set shape:", X_val.shape, y_val.shape)

    return X_train, X_val, y_train, y_val, label_encoder
