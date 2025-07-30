import pandas as pd
import numpy as np

class BaseMMMModel:
    def __init__(self, adstock_decay=0.0):
        self.adstock_decay = adstock_decay
        self.is_trained = False

    def apply_adstock(self, spend_series, decay_rate):
        #geometric adstock transformation using vectorized operations
        if decay_rate <= 0:
            return spend_series.values
        
        #convert to numpy for faster computation
        spend_values = spend_series.values
        adstocked = np.zeros_like(spend_values, dtype=np.float64)
        
        #vectorized geometric adstock using numpy
        adstocked[0] = spend_values[0]
        for i in range(1, len(spend_values)):
            adstocked[i] = spend_values[i] + decay_rate * adstocked[i-1]
        
        return adstocked

    def apply_log_transform(self, series_or_array):
        #log transformation for diminishing returns
        if hasattr(series_or_array, 'values'):
            return np.log1p(series_or_array.values)
        return np.log1p(series_or_array)

    def add_features(self, df, spend_cols):
        #create adstock and log transformed features efficiently
        result_df = df.copy()
        
        #prepare arrays for batch operations
        feature_data = {}
        
        #adstock for spend columns only
        if self.adstock_decay > 0:
            for col in spend_cols:
                if col in df.columns:
                    adstocked = self.apply_adstock(df[col], self.adstock_decay)
                    feature_data[f"{col}_adstock"] = adstocked

        #log transform all numeric columns
        numeric_cols = [col for col in df.columns if col != 'date' and df[col].dtype in ['float64', 'int64']]
        for col in numeric_cols:
            log_values = self.apply_log_transform(df[col].values)
            feature_data[f"{col}_log"] = log_values
        
        #log transform adstocked features (reuse calculated adstock)
        if self.adstock_decay > 0:
            for col in spend_cols:
                if col in df.columns:
                    adstock_key = f"{col}_adstock"
                    if adstock_key in feature_data:
                        log_adstocked = self.apply_log_transform(feature_data[adstock_key])
                        feature_data[f"{col}_adstock_log"] = log_adstocked

        #batch create new columns
        for feature_name, feature_values in feature_data.items():
            result_df[feature_name] = feature_values

        return result_df

    #to be implemented in subclasses
    def train(self, df, sales_col, spend_cols): raise NotImplementedError
    def predict(self, data): raise NotImplementedError 