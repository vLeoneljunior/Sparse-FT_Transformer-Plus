import numpy as np
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_california_data():
    """Charge et prépare les données California Housing"""
    data = fetch_california_housing()
    X = data.data
    y = data.target
    
    # Normalisation
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y

def get_data_splits(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """Divise les données en ensembles train, validation et test"""
    # Split initial: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Split secondaire: train vs val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state
    )
    
    # Conversion en tenseurs PyTorch
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    return {
        'X_train': X_train_tensor,
        'X_val': X_val_tensor,
        'X_test': X_test_tensor,
        'y_train': y_train_tensor,
        'y_val': y_val_tensor,
        'y_test': y_test_tensor,
        'n_features': X_train.shape[1]
    }