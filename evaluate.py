import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data_processing import load_dataset, preprocess_data
import os

# Paths
DATA_PATH = r"C:\Users\satya\DON_Prediction\data\MLE-Assignment.csv"
MODEL_PATH = r"C:\Users\satya\DON_Prediction\models\don_model.h5"

def main():
    # Load and preprocess data
    df = load_dataset(DATA_PATH)

    print("🧾 Columns:", df.columns.tolist())  # Show column names for debug

    # Correct target column name
    target_col = "vomitoxin_ppb"
    if target_col not in df.columns:
        raise ValueError(f"❌ Column '{target_col}' not found in dataset.")

    X, y, _ = preprocess_data(df)
    feature_names = [col for col in df.columns if col != target_col and col != 'hsi_id']

    # Load model
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # Predict
    y_pred = model.predict(X)

    # Evaluate
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    print(f"📊 MAE:  {mae:.2f}")
    print(f"📊 RMSE: {rmse:.2f}")
    print(f"📊 R²:   {r2:.2f}")

    # LIME explanation
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X,
        feature_names=feature_names,
        mode='regression'
    )

    sample_idx = 0
    exp = explainer.explain_instance(
        data_row=X[sample_idx],
        predict_fn=model.predict
    )

    os.makedirs("outputs", exist_ok=True)
    exp.save_to_file("outputs/lime_explanation_sample0.html")
    print("✅ LIME explanation saved to: outputs/lime_explanation_sample0.html")

    # Scatter Plot
    plt.figure(figsize=(6, 5))
    plt.scatter(y, y_pred, alpha=0.6)
    plt.xlabel("Actual vomitoxin_ppb")
    plt.ylabel("Predicted vomitoxin_ppb")
    plt.title("Actual vs Predicted DON Concentration")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/actual_vs_predicted.png")
    plt.show()

if __name__ == "__main__":
    main()
