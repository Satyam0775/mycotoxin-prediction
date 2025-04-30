import sys
import os
import streamlit as st
import pandas as pd
import tensorflow as tf

# Add src/ to the Python path so Streamlit can find the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import preprocess_data

# Path to the pre-trained model
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'don_model.h5'))

# Load model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Streamlit UI configuration
st.set_page_config(page_title="DON Predictor", layout="wide")
st.title("🌽 DON Concentration Predictor (Mycotoxin in Corn)")
st.markdown("Upload **hyperspectral CSV data** to predict **Vomitoxin (DON)** levels (in ppb).")

# File uploader
uploaded_file = st.file_uploader("📁 Upload your CSV File", type="csv")

if uploaded_file is not None:
    try:
        # Read uploaded file
        df = pd.read_csv(uploaded_file)

        # Show preview of uploaded data
        st.subheader("📊 Uploaded Data Preview:")
        st.dataframe(df.head())

        # Preprocess input data
        X_scaled, _, _ = preprocess_data(df)

        # Predict DON concentration
        predictions = model.predict(X_scaled).flatten()
        df['Predicted_DON_ppb'] = predictions

        # Display results
        st.success("✅ Prediction complete!")
        st.subheader("🔬 Predicted DON Concentrations (First 10 Rows):")
        st.dataframe(df.head(10))

        # Download predictions as CSV
        csv = df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Predictions as CSV",
            data=csv,
            file_name="don_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")

else:
    st.info("📄 Please upload a CSV file with spectral reflectance features.")

# Footer
st.caption("🧠 Model trained on hyperspectral reflectance data to predict DON concentration (ppb).")
