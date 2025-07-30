import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
from .base_model import BaseMMMModel

class BayesianMMMModel(BaseMMMModel):
    def __init__(self, adstock_decay=0.0):
        super().__init__(adstock_decay)
        self.trace = None
        self.alpha_mean = None
        self.betas_mean = None

    def train(self, df, sales_col, spend_cols, draws=500):
        #add transformed features
        df_features = self.add_features(df, spend_cols)
        
        #prepare training data
        self.feature_cols = [col for col in df_features.columns 
                           if col != sales_col and col != 'date' and 
                           any(s in col for s in spend_cols)]
        
        X = df_features[self.feature_cols].values
        y = df_features[sales_col].values
        
        #bayesian linear regression
        with pm.Model() as model:
            alpha = pm.Normal('alpha', mu=y.mean(), sigma=y.std())
            betas = pm.Normal('betas', mu=0, sigma=1, shape=len(self.feature_cols))
            sigma = pm.HalfNormal('sigma', sigma=y.std())
            
            mu = alpha + pm.math.dot(X, betas)
            pm.Normal('y', mu=mu, sigma=sigma, observed=y)
            
            #sample posterior
            self.trace = pm.sample(
                draws=draws,
                tune=1000,
                chains=2,
                return_inferencedata=True,
                target_accept=0.85
            )
        
        #store posterior means for fast prediction
        self.alpha_mean = self.trace.posterior['alpha'].mean().values
        self.betas_mean = self.trace.posterior['betas'].mean(dim=['chain', 'draw']).values
        
        self.spend_cols = spend_cols
        self.is_trained = True
        print(f"Bayesian sampling complete ({draws} draws)")
    
    def predict(self, data):
        #dynamically derive spend columns from data
        spend_cols = [col for col in data.columns if col.endswith('_spend')]
        
        #make predictions using posterior means
        df_features = self.add_features(data, spend_cols)
        feature_cols = [col for col in df_features.columns 
                       if col not in ['sales', 'date'] and 
                       any(s in col for s in spend_cols)]
        X_new = df_features[feature_cols].values
        predictions = self.alpha_mean + np.dot(X_new, self.betas_mean)
        return predictions
    
    def save_trace(self, filepath):
        #save trace for later use
        if self.trace is not None:
            self.trace.to_netcdf(filepath)
    
    def load_trace(self, filepath):
        #load saved trace
        self.trace = az.from_netcdf(filepath)
        if self.trace is not None:
            self.alpha_mean = self.trace.posterior['alpha'].mean().values
            self.betas_mean = self.trace.posterior['betas'].mean(dim=['chain', 'draw']).values
            self.is_trained = True