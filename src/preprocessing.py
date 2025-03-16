import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """ Load dataset from a CSV file """
    return pd.read_csv(filepath)

def preprocess_data(df, target_column):
    """ Preprocess data by handling missing values and scaling """
    
    # Drop rows with missing target values
    df = df.dropna(subset=[target_column])  # Use the target_column passed as argument
    
    # Fill missing numerical values with mean of that column
    df = df.fillna(df.mean())
    
    # Split features (X) and target (y)
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Feature scaling: Apply StandardScaler to features only (not target)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X_scaled, y

def split_data(X, y, test_size=0.2, random_state=42):
    """ Split data into training and testing sets """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Example Usage
if __name__ == "__main__":
    # Example file path to your dataset
    file_path = "your_dataset.csv"  # Change this to the correct file path
    
    # Load data
    df = load_data(file_path)
    
    # Define the target column (replace 'target_column' with the actual column name)
    target_column = 'target_column'  # Change this to the actual target column name
    
    # Preprocess data
    X_scaled, y = preprocess_data(df, target_column)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    
    # Now you can proceed with your model training using X_train, X_test, y_train, y_test
