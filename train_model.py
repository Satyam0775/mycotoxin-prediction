import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src.data_processing import load_dataset, preprocess_data

# Constants
DATA_PATH = r"C:\Users\satya\DON_Prediction\data\MLE-Assignment.csv"
MODEL_PATH = r"C:\Users\satya\DON_Prediction\models\don_model.h5"

def build_model(input_dim: int) -> tf.keras.Model:
    """
    Build and return a regression neural network.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def main():
    # Load and preprocess data
    df = load_dataset(DATA_PATH)
    X, y, _ = preprocess_data(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build model
    model = build_model(X.shape[1])

    # Train model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"✅ Model saved at: {MODEL_PATH}")

if __name__ == "__main__":
    main()
