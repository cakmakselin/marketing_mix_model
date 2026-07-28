import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from .base_model import BaseMMMModel


class LinearMMMModel(BaseMMMModel):
    def __init__(self, adstock_decay=0.0):
        super().__init__(adstock_decay)
        self.model = LinearRegression()

    def train(self, df, sales_col, spend_cols):
        # build spend-derived features (never includes sales)
        X = self.build_features(df, spend_cols)
        y = df[sales_col]

        with np.errstate(all='ignore'):
            self.model.fit(X, y)
            r2 = self.model.score(X, y)
        self.feature_cols = list(X.columns)
        self.spend_cols = list(spend_cols)
        self.is_trained = True

        print(f"In-sample R² = {r2:.3f} (see holdout metrics for real performance)")

    def predict(self, data):
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        # use the channels the model was trained on so coefficients stay aligned
        X_new = self.build_features(data, self.spend_cols)[self.feature_cols]
        with np.errstate(all='ignore'):  # see comment in train()
            predictions = self.model.predict(X_new)
        if not np.isfinite(predictions).all():
            raise FloatingPointError("Linear model produced non-finite predictions")
        return predictions

    def raw_coefficients(self):
        return float(self.model.intercept_), self.model.coef_

    def save(self, filepath):
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
        with open(filepath, 'rb') as f:
            payload = pickle.load(f)

        if not isinstance(payload, dict):
            raise ValueError(
                f"Unsupported model file format in {filepath}; "
                "retrain with scripts/train_models.py"
            )

        self.model = payload["model"]
        self.adstock_decay = payload["adstock_decay"]
        self.feature_cols = payload["feature_cols"]
        self.spend_cols = payload["spend_cols"]
        self.is_trained = payload["is_trained"]
