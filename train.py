# train.py

import os
import joblib
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from prep import prepare_data  # Import the data preparation function

def main():
    # Set absolute paths (escaped backslashes for Windows)
    csv_path = r'C:\Users\Kushal Vishwakarma\Downloads\car_damage\car_damage\train\train.csv'
    images_dir = r'C:\Users\Kushal Vishwakarma\Downloads\car_damage\car_damage\train\images'

    # Prepare data
    X_train, X_val, y_train, y_val, label_encoder = prepare_data(csv_path, images_dir)

    # Save label encoder for later use
    joblib.dump(label_encoder, "label_encoder.pkl")

    # Build model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze base model

    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(label_encoder.classes_), activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint("best_model.h5", save_best_only=True)
    ]

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=10,
        validation_data=(X_val, y_val),
        batch_size=64,
        callbacks=callbacks
    )

    # Evaluate the model
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=2)
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

    # Save the final model
    model.save("car_damage_model_final.h5")


if __name__ == "__main__":
    main()
