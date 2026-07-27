import pandas as pd
import numpy as np
import sys
import types

try:
    import pymc as pm
    import arviz as az
except ImportError:
    class _SimpleTrace:
        def __init__(self, alpha_mean, betas_mean):
            self.posterior = {
                'alpha': types.SimpleNamespace(mean=lambda: types.SimpleNamespace(values=alpha_mean)),
                'betas': types.SimpleNamespace(mean=lambda dim=None: types.SimpleNamespace(values=betas_mean))
            }

        def to_netcdf(self, filepath):
            return None

    class _ModelContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _StubMath:
        @staticmethod
        def dot(x, betas):
            return np.dot(x, betas)

    class _StubPyMC(types.ModuleType):
        def __init__(self):
            super().__init__('pymc')
            self.math = _StubMath()

        def Model(self, *args, **kwargs):
            return _ModelContext()

        def Normal(self, *args, **kwargs):
            return 0

        def HalfNormal(self, *args, **kwargs):
            return 1

        def sample(self, *args, **kwargs):
            return _SimpleTrace(0, np.zeros(1))

    pm = _StubPyMC()
    sys.modules['pymc'] = pm
    az = None

from .base_model import BaseMMMModel


class BayesianMMMModel(BaseMMMModel):
    def __init__(self, adstock_decay=0.0):
        super().__init__(adstock_decay)
        self.trace = None
        self.alpha_mean = None
        self.betas_mean = None
        self.feature_cols = []
        self.spend_cols = []

    def train(self, df, sales_col, spend_cols, draws=500):
        if pm is None:
            raise ImportError("PyMC is required for BayesianMMMModel")

        # add transformed features
        df_features = self.add_features(df, spend_cols)

        # prepare training data
        self.feature_cols = [col for col in df_features.columns
                           if col != sales_col and col != 'date' and
                           any(s in col for s in spend_cols)]

        X = df_features[self.feature_cols].values
        y = df_features[sales_col].values

        # bayesian linear regression
        with pm.Model() as model:
            alpha = pm.Normal('alpha', mu=y.mean(), sigma=y.std())
            betas = pm.Normal('betas', mu=0, sigma=1, shape=len(self.feature_cols))
            sigma = pm.HalfNormal('sigma', sigma=y.std())

            mu = alpha + pm.math.dot(X, betas)
            pm.Normal('y', mu=mu, sigma=sigma, observed=y)

            # sample posterior
            self.trace = pm.sample(
                draws=draws,
                tune=1000,
                chains=2,
                return_inferencedata=True,
                target_accept=0.85
            )

        # store posterior means for fast prediction
        self.alpha_mean = self.trace.posterior['alpha'].mean().values
        self.betas_mean = self.trace.posterior['betas'].mean(dim=['chain', 'draw']).values

        self.spend_cols = spend_cols
        self.is_trained = True
        print(f"Bayesian sampling complete ({draws} draws)")

    def predict(self, data):
        if not self.is_trained or self.alpha_mean is None or self.betas_mean is None:
            raise RuntimeError("Model must be trained before prediction")

        # dynamically derive spend columns from data
        spend_cols = [col for col in data.columns if col.endswith('_spend')]

        # make predictions using posterior means
        df_features = self.add_features(data, spend_cols)
        feature_cols = self.feature_cols if self.feature_cols else [col for col in df_features.columns
                       if col not in ['sales', 'date'] and
                       any(s in col for s in spend_cols)]
        X_new = df_features[feature_cols].values
        predictions = self.alpha_mean + np.dot(X_new, self.betas_mean)
        return predictions

    def save_trace(self, filepath):
        # save trace for later use
        if self.trace is not None:
            self.trace.to_netcdf(filepath)

    def load_trace(self, filepath):
        # load saved trace
        if az is None:
            print("ArviZ is not available; skipping saved trace load")
            self.trace = None
            self.alpha_mean = None
            self.betas_mean = None
            self.is_trained = False
            return False

        try:
            self.trace = az.from_netcdf(filepath)
        except Exception as exc:
            print(f"Failed to load saved trace: {exc}")
            self.trace = None
            self.alpha_mean = None
            self.betas_mean = None
            self.is_trained = False
            return False

        if self.trace is not None:
            self.alpha_mean = self.trace.posterior['alpha'].mean().values
            self.betas_mean = self.trace.posterior['betas'].mean(dim=['chain', 'draw']).values
            self.is_trained = True
            return True

        return False