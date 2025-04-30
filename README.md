# 🌽 DON Concentration Prediction App (Mycotoxin Detection)

A machine learning-based pipeline to predict **Deoxynivalenol (DON)** concentration in corn samples using **hyperspectral imaging data**. This project leverages neural networks to forecast DON (vomitoxin) levels and provides an interactive web application for real-time predictions.

🚀 [Live App on Hugging Face Spaces](https://huggingface.co/spaces/Satyam0077/don-prediction-app)

---

## 📁 Project Structure

```bash
don-prediction-app/
├── api/                      # Streamlit frontend (main.py)
│   └── app.py
├── data/                    # Input dataset
│   └── MLE-Assignment.csv
├── models/                  # Trained Keras model
│   └── don_model.h5
├── notebooks/               # Jupyter notebook for experimentation
│   └── model_training.ipynb
├── outputs/                 # Visual outputs
│   ├── actual_vs_predicted.png
│   └── lime_explanation_sample0.html
├── src/                     # Modular source code
│   ├── data_processing.py
│   ├── evaluate.py
│   └── train_model.py
├── requirements.txt         # Python dependencies
├── README.md                # Project overview and instructions
└── Short_Report_DON_Prediction.pdf  # Final report

Problem Statement
Mycotoxins like DON pose serious threats to food safety. This project aims to build a predictive model using hyperspectral reflectance data to estimate DON contamination in corn.

✅ Key Features
Data preprocessing with normalization and cleaning

Model training using a deep neural network (DNN)

Evaluation with MAE, RMSE, and R² metrics

LIME interpretability for model explanation

Streamlit-based app for interactive DON prediction

Deployable pipeline with modular code and reusable components

🧪 How to Run Locally
1. Clone the repository
bash
Copy
Edit
git clone https://github.com/yourusername/don-prediction-app.git
cd don-prediction-app
2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Run the Streamlit App
bash
Copy
Edit
cd api
streamlit run app.py
🛠️ Model Details
Architecture: Feedforward Neural Network

Input: 448 spectral features

Hidden Layers: 128 and 64 neurons (ReLU)

Output: Single regression value (DON ppb)

Loss: Mean Absolute Error (MAE)

Optimizer: Adam

Epochs: 50

Performance:

MAE: ~4018.47

RMSE: ~13164.73

R²: -0.01

🧠 Explainability
LIME: Used to highlight features that most influence individual predictions

Visual explanations saved in outputs/lime_explanation_sample0.html

📈 Results Snapshot


📄 Final Report
Detailed analysis, methodology, and evaluation results are provided in:

Short_Report_DON_Prediction.pdf

🙋 Author
Satyam Kumar

📜 License
This project is released under the MIT License. Feel free to reuse and contribute!

