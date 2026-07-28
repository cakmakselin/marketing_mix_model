import pandas as pd
import numpy as np
from typing import Dict

def calculate_mape(actual: pd.Series, predicted: pd.Series) -> float:
    #calculate mean absolute percentage error
    if len(actual) != len(predicted):
        raise ValueError("Series must have same length")
    
    #exclude zero actuals to avoid division by zero
    mask = np.asarray(actual) != 0
    if not mask.any():
        raise ValueError("MAPE undefined: all actual values are zero")
    
    actual_filtered = actual[mask]
    predicted_filtered = predicted[mask]
    
    #calculate mape
    percentage_errors = np.abs((actual_filtered - predicted_filtered) / actual_filtered)
    mape = np.mean(percentage_errors) * 100
    
    return float(mape)

def calculate_r2(actual: pd.Series, predicted: pd.Series) -> float:
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return float(r2)

def evaluate_model(actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
    #evaluate model with multiple metrics
    return {
        'mape': calculate_mape(actual, predicted),
        'r2': calculate_r2(actual, predicted)
    } 