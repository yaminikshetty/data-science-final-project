# train_asl_model.py
# =======================================================
# Improved CNN model for American Sign Language (A–Y)
# Dataset: sign_mnist_train.csv and sign_mnist_test.csv
# =======================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# -------------------------------
# Configuration
# -------------------------------
TRAIN_PATH = "sign_mnist_train.csv"
TEST_PATH = "sign_mnist_test.csv"
MODEL_SAVE_PATH = "improved_asl_model.h5"
NUM_CLASSES = 25  # A-Y (J & Z excluded)
EPOCHS = 25
BATCH_SIZE = 128
TEST_SIZE = 0.2
RANDOM_STATE = 42

# -------------------------------
# Load dataset
# -------------------------------
def load_dataset(train_path, test_path):
    print("📂 Loading dataset...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"✅ Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    return train_df, test_df

# -------------------------------
# Prepare data
# -------------------------------
def prepare_data(train_df, test_df):
    print("\n🔄 Preparing data...")

    X = train_df.drop('label', axis=1).values
    y = train_df['label'].values

    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values

    X = X.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    X = X.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    y = to_categorical(y, NUM_CLASSES)
    y_test = to_categorical(y_test, NUM_CLASSES)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"✅ Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test

# -------------------------------
# Build improved CNN model
# -------------------------------
def build_model(num_classes):
    print("\n🧠 Building CNN model...")

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        Dropout(0.4),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model

# -------------------------------
# Train model with data augmentation
# -------------------------------
def train_model(model, X_train, y_train, X_val, y_val):
    print("\n🚀 Starting training...")

    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    datagen.fit(X_train)

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        verbose=1
    )

    print("✅ Training complete!")
    return history

# -------------------------------
# Evaluate and save
# -------------------------------
def evaluate_and_save(model, X_val, y_val, X_test, y_test):
    print("\n📊 Evaluating model...")
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    print(f"✅ Validation Accuracy: {val_acc*100:.2f}%")
    print(f"✅ Test Accuracy: {test_acc*100:.2f}%")

    print(f"\n💾 Saving model to '{MODEL_SAVE_PATH}'...")
    model.save(MODEL_SAVE_PATH)
    print("✅ Model saved successfully!")

# -------------------------------
# Plot training history
# -------------------------------
def plot_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

# -------------------------------
# Main
# -------------------------------
def main():
    print("="*60)
    print("ASL SIGN LANGUAGE - IMPROVED CNN TRAINING")
    print("="*60)

    train_df, test_df = load_dataset(TRAIN_PATH, TEST_PATH)
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(train_df, test_df)

    model = build_model(NUM_CLASSES)
    history = train_model(model, X_train, y_train, X_val, y_val)
    evaluate_and_save(model, X_val, y_val, X_test, y_test)
    plot_history(history)

    print("\n🎯 Training complete and model ready for prediction!")

if __name__ == "__main__":
    main()
asl