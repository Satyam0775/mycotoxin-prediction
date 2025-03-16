import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify

# Load trained model
MODEL_PATH = r"C:\Users\satya\mycotoxin\mycotoxin_prediction\deployment\mycotoxin_model_fixed.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Initialize Flask App
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "DON Concentration Prediction API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        spectral_values = np.array(data["spectral_values"]).reshape(1, -1)
        
        prediction = model.predict(spectral_values)
        return jsonify({"predicted_don": float(prediction[0][0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  # Run on all interfaces
