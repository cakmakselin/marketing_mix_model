import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from .base_model import BaseMMMModel


class LinearMMMModel(BaseMMMModel):
    def __init__(self, adstock_decay=0.0):
        super().__init__(adstock_decay)
        self.model = LinearRegression()
        self.feature_cols = []
        self.spend_cols = []

    def train(self, df, sales_col, spend_cols):
        # add transformed features
        df_features = self.add_features(df, spend_cols)

        # prepare training data
        self.feature_cols = [col for col in df_features.columns if col not in [sales_col, 'date']]
        X = df_features[self.feature_cols]
        y = df_features[sales_col]

        # train model
        self.model.fit(X, y)
        self.spend_cols = spend_cols
        self.is_trained = True

        # show performance
        r2 = self.model.score(X, y)
        print(f"R² = {r2:.3f}")

    def predict(self, data):
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        # dynamically derive spend columns from data
        spend_cols = [col for col in data.columns if col.endswith('_spend')]

        # make predictions
        df_features = self.add_features(data, spend_cols)
        feature_cols = self.feature_cols if self.feature_cols else [col for col in df_features.columns if col not in ['sales', 'date']]
        X_new = df_features[feature_cols]
        predictions = self.model.predict(X_new)
        return predictions

    def save(self, filepath):
        # save the trained sklearn model and metadata
        payload = {
            "model": self.model,
            "adstock_decay": self.adstock_decay,
            "feature_cols": self.feature_cols,
            "spend_cols": self.spend_cols,
            "is_trained": self.is_trained,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(payload, f)

    def load(self, filepath):
        # load sklearn model and metadata
        with open(filepath, 'rb') as f:
            payload = pickle.load(f)

        if isinstance(payload, dict):
            self.model = payload.get("model", self.model)
            self.adstock_decay = payload.get("adstock_decay", self.adstock_decay)
            self.feature_cols = payload.get("feature_cols", [])
            self.spend_cols = payload.get("spend_cols", [])
            self.is_trained = payload.get("is_trained", True)
        else:
            self.model = payload
            self.is_trained = True
