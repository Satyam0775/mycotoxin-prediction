import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load the hyperspectral dataset from a CSV file.
    """
    return pd.read_csv(file_path)


def preprocess_data(df: pd.DataFrame) -> Tuple[np.ndarray, pd.Series, MinMaxScaler]:
    """
    Preprocess the data: drop ID, handle missing values, normalize features, return target.
    """
    df = df.drop(columns=['hsi_id'], errors='ignore')
    df.fillna(df.mean(), inplace=True)

    X = df.drop(columns=['vomitoxin_ppb'])
    y = df['vomitoxin_ppb']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


def check_data_quality(df: pd.DataFrame) -> None:
    """
    Print checks for missing values, shape, and summary stats.
    """
    print("📦 Data Shape:", df.shape)
    print("🧪 Missing Values:", df.isnull().sum().sum())
    print(df.describe())
