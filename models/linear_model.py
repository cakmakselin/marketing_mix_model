import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from .base_model import BaseMMMModel

class LinearMMMModel(BaseMMMModel):
    def __init__(self, adstock_decay=0.0):
        super().__init__(adstock_decay)
        self.model = LinearRegression()

    def train(self, df, sales_col, spend_cols):
        #add transformed features
        df_features = self.add_features(df, spend_cols)
        
        #prepare training data
        self.feature_cols = [col for col in df_features.columns if col not in [sales_col, 'date']]
        X = df_features[self.feature_cols]
        y = df_features[sales_col]
        
        #train model
        self.model.fit(X, y)
        self.spend_cols = spend_cols
        self.is_trained = True
        
        #show performance
        r2 = self.model.score(X, y)
        print(f"R² = {r2:.3f}")

    def predict(self, data):
        spend_cols = [col for col in data.columns if col.endswith('_spend')]
        
        #make predictions
        df_features = self.add_features(data, spend_cols)
        feature_cols = [col for col in df_features.columns if col not in ['sales', 'date']]
        X_new = df_features[feature_cols] 
        predictions = self.model.predict(X_new)
        return predictions
    
    def save(self, filepath):
        #save only the trained sklearn model
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, filepath):
        #load sklearn model
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True
