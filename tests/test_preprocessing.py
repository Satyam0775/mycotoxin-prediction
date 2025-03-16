import pandas as pd
from src.preprocessing import preprocess_data

def test_preprocessing():
    # Load the dataset
    df = pd.read_csv(r"C:\Users\satya\mycotoxin\mycotoxin_prediction\data\cleaned_dataset.csv")
    
    # Preprocess the data
    df_cleaned = preprocess_data(df)
    
    # Check if there are any missing values in the cleaned data
    missing_values = df_cleaned.isnull().sum().sum()
    assert missing_values == 0, f"Test failed: There are {missing_values} missing values in the cleaned data."
    
    # Check for expected column names in cleaned data
    assert 'feature_column' in df_cleaned.columns, "Test failed: 'feature_column' is missing."
    
    print("Test passed: No missing values in the cleaned dataset.")

# Call the test function
test_preprocessing()
