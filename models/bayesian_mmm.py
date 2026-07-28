import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .base_model import BaseMMMModel


def _import_pymc():
    #lazy import so the linear model works without PyMC installed,
    #but a broken/missing PyMC fails loudly instead of silently faking results
    try:
        import pymc as pm
        return pm
    except ImportError as exc:
        raise ImportError(
            "PyMC is required for BayesianMMMModel but could not be imported. "
            "Install a compatible environment (Python >= 3.10, see requirements.txt). "
            f"Original error: {exc}"
        ) from exc


def _import_arviz():
    try:
        import arviz as az
        return az
    except ImportError as exc:
        raise ImportError(
            "ArviZ is required to save/load Bayesian traces. "
            f"Original error: {exc}"
        ) from exc


class BayesianMMMModel(BaseMMMModel):
    def __init__(self, adstock_decay=0.0):
        super().__init__(adstock_decay)
        self.trace = None
        self.alpha_mean = None
        self.betas_mean = None
        #feature standardization parameters, fixed at training time
        self.feature_means = None
        self.feature_stds = None

    def _standardize(self, X):
        return (X - self.feature_means) / self.feature_stds

    def train(self, df, sales_col, spend_cols, draws=500):
        pm = _import_pymc()

        X_df = self.build_features(df, spend_cols)
        self.feature_cols = list(X_df.columns)
        self.spend_cols = list(spend_cols)

        #standardize features so priors are on a common, known scale
        self.feature_means = X_df.mean().values
        self.feature_stds = X_df.std().replace(0, 1.0).values
        X = self._standardize(X_df.values)
        y = df[sales_col].values

        n_spend = len(spend_cols)
        with pm.Model():
            alpha = pm.Normal('alpha', mu=y.mean(), sigma=y.std())
            #spend effects constrained positive: marketing should not reduce sales
            betas_spend = pm.HalfNormal('betas_spend', sigma=y.std(), shape=n_spend)
            #seasonal effects can go either way
            betas_season = pm.Normal('betas_season', mu=0, sigma=y.std(),
                                     shape=len(self.feature_cols) - n_spend)
            sigma = pm.HalfNormal('sigma', sigma=y.std())

            betas = pm.math.concatenate([betas_spend, betas_season])
            mu = alpha + pm.math.dot(X, betas)
            pm.Normal('y', mu=mu, sigma=sigma, observed=y)

            self.trace = pm.sample(
                draws=draws,
                tune=1000,
                chains=2,
                return_inferencedata=True,
                target_accept=0.9,
                progressbar=False,
            )

        #report convergence instead of assuming it
        az = _import_arviz()
        summary = az.summary(self.trace, var_names=['alpha', 'betas_spend', 'betas_season', 'sigma'])
        divergences = int(self.trace.sample_stats['diverging'].values.sum())
        max_rhat = float(summary['r_hat'].max())
        print(f"Sampling done: {draws} draws | divergences={divergences} | max r_hat={max_rhat:.3f}")
        if divergences > 0 or max_rhat > 1.05:
            print("WARNING: sampling diagnostics look problematic; treat estimates with caution")

        self.alpha_mean = float(self.trace.posterior['alpha'].mean().values)
        self.betas_mean = self._posterior_betas().mean(axis=0)
        self.is_trained = True

    def _posterior_betas(self):
        #stack spend and season betas into draws x features, matching feature_cols order
        post = self.trace.posterior
        spend = post['betas_spend'].values.reshape(-1, post['betas_spend'].shape[-1])
        season = post['betas_season'].values.reshape(-1, post['betas_season'].shape[-1])
        return np.concatenate([spend, season], axis=1)

    def _design_matrix(self, data):
        X_df = self.build_features(data, self.spend_cols)[self.feature_cols]
        return self._standardize(X_df.values)

    def predict(self, data):
        if not self.is_trained or self.alpha_mean is None or self.betas_mean is None:
            raise RuntimeError("Model must be trained before prediction")
        X = self._design_matrix(data)
        return self.alpha_mean + np.dot(X, self.betas_mean)

    def raw_coefficients(self):
        #posterior mean coefficients converted back from the standardized scale
        coefs = self.betas_mean / self.feature_stds
        intercept = self.alpha_mean - float(np.sum(self.betas_mean * self.feature_means / self.feature_stds))
        return intercept, coefs

    def predict_interval(self, data, hdi_prob=0.9):
        #credible interval of expected sales from posterior draws (needs the trace)
        if self.trace is None:
            return None
        X = self._design_matrix(data)
        alpha_draws = self.trace.posterior['alpha'].values.reshape(-1)
        betas_draws = self._posterior_betas()
        #np.dot instead of `@`: the matmul operator triggers spurious FP
        #warnings on macOS Accelerate BLAS builds
        preds = alpha_draws[None, :] + np.dot(X, betas_draws.T)  # (rows, draws)
        lo = (1 - hdi_prob) / 2 * 100
        return np.percentile(preds, [lo, 100 - lo], axis=1)

    def save(self, trace_path):
        trace_path = Path(trace_path)
        if self.trace is not None:
            tmp_path = trace_path.with_suffix('.nc.tmp')
            self.trace.to_netcdf(str(tmp_path))
            os.replace(tmp_path, trace_path)
        meta = {
            "adstock_decay": self.adstock_decay,
            "feature_cols": self.feature_cols,
            "spend_cols": self.spend_cols,
            "feature_means": np.asarray(self.feature_means).tolist(),
            "feature_stds": np.asarray(self.feature_stds).tolist(),
            "alpha_mean": self.alpha_mean,
            "betas_mean": np.asarray(self.betas_mean).tolist(),
        }
        meta_path = self._meta_path(trace_path)
        tmp_meta = meta_path.with_suffix('.json.tmp')
        with open(tmp_meta, 'w') as f:
            json.dump(meta, f, indent=2)
        os.replace(tmp_meta, meta_path)

    def load(self, trace_path):
        meta_path = self._meta_path(trace_path)
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Model metadata not found: {meta_path}. Retrain with scripts/train_models.py"
            )
        with open(meta_path) as f:
            meta = json.load(f)

        self.adstock_decay = meta["adstock_decay"]
        self.feature_cols = meta["feature_cols"]
        self.spend_cols = meta["spend_cols"]
        self.feature_means = np.array(meta["feature_means"])
        self.feature_stds = np.array(meta["feature_stds"])
        self.alpha_mean = meta["alpha_mean"]
        self.betas_mean = np.array(meta["betas_mean"])
        self.is_trained = True

        #trace is optional: predictions work from posterior means alone
        trace_path = Path(trace_path)
        if trace_path.exists():
            az = _import_arviz()
            self.trace = az.from_netcdf(str(trace_path))
            self.trace.load()
            self.trace.close()
        else:
            self.trace = None
        return True

    @staticmethod
    def _meta_path(trace_path):
        return Path(trace_path).with_suffix('.json')
