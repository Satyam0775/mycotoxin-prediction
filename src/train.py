# src/train.py
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
from src.logging_config import setup_logger

# Initialize logger
logger = setup_logger()

def load_data(file_path):
    """ Function to load data from CSV file. """
    logger.info(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info("Data loaded successfully.")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def train_model(df):
    """ Function to train the model. """
    logger.info("Model training started.")
    try:
        # Preprocess the data
        X = df.drop('target', axis=1)  # Replace 'target' with your actual target column name
        y = df['target']  # Replace 'target' with your actual target column name
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize and train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        logger.info("Model training completed.")
        
        # Return the trained model
        return model
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise

def main():
    """ Main function to execute the pipeline. """
    try:
        # Load and preprocess data
        df = load_data('path_to_your_data.csv')
        
        # Train the model
        model = train_model(df)
        
        # Optionally, you can save the model or evaluate it here
        logger.info("Model is ready.")
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")

if __name__ == '__main__':
    main()
