#!/usr/bin/env python3
"""Train both MMM models with a train/validation/test protocol.

- Validation: expanding-window time series CV on the first 80% of days,
  used to tune the adstock decay (the model's only hyperparameter).
  Model/feature choices are made here, never on the test window.
- Test: the untouched last 20% of days, evaluated exactly once per model.
- Serving artifacts: refit on all data with the tuned decay.
"""

import contextlib
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.linear_model import LinearMMMModel
from models.bayesian_mmm import BayesianMMMModel
from data.ingestion import DataIngestor
from evaluation.metrics import evaluate_model, calculate_mape
from config import config

DECAY_GRID = [round(d, 1) for d in np.arange(0.0, 0.8, 0.1)]
N_FOLDS = 3


def tune_adstock_decay(pool_df, spend_cols):
    """Pick the adstock decay by average MAPE over expanding-window CV folds.

    Tuning uses the linear model because it is fast; both models share the
    same feature transform, so the tuned decay transfers to the Bayesian one.
    """
    splitter = TimeSeriesSplit(n_splits=N_FOLDS)
    cv_mape = {}
    for decay in DECAY_GRID:
        fold_mapes = []
        for train_idx, val_idx in splitter.split(pool_df):
            model = LinearMMMModel(adstock_decay=decay)
            with contextlib.redirect_stdout(io.StringIO()):  # silence per-fold prints
                model.train(pool_df.iloc[train_idx], 'sales', spend_cols)
            val_fold = pool_df.iloc[val_idx]
            predictions = model.predict(val_fold)
            fold_mapes.append(calculate_mape(val_fold['sales'], predictions))
        cv_mape[decay] = round(float(np.mean(fold_mapes)), 3)

    best_decay = min(cv_mape, key=cv_mape.get)
    return best_decay, cv_mape


def test_metrics(model_cls, pool_df, test_df, spend_cols, decay):
    #fit on the full train+val pool, evaluate once on the untouched test window
    model = model_cls(adstock_decay=decay)
    model.train(pool_df, 'sales', spend_cols)
    predictions = model.predict(test_df)
    return evaluate_model(test_df['sales'], predictions)


def train_models():
    print("Loading data...")
    ingestor = DataIngestor()
    data = ingestor.run(training=True)

    spend_cols = [col for col in data.columns if col.endswith('_spend')]
    data = data[['date', 'sales'] + spend_cols].copy()

    #temporal split: first 80% is the train+validation pool, last 20% is test
    split = int(len(data) * 0.8)
    pool_df, test_df = data.iloc[:split], data.iloc[split:]
    print(f"\nData: {data.shape}, channels: {spend_cols}")
    print(f"Train+val pool: {pool_df['date'].min().date()}..{pool_df['date'].max().date()} "
          f"| test (untouched): {test_df['date'].min().date()}..{test_df['date'].max().date()}")

    print(f"\nTuning adstock decay via expanding-window CV ({N_FOLDS} folds) on the pool...")
    best_decay, cv_mape = tune_adstock_decay(pool_df, spend_cols)
    print("Validation MAPE by decay:", cv_mape)
    print(f"Chosen adstock_decay: {best_decay}")

    models_dir = config.saved_models_path
    models_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "trained_at": datetime.now().isoformat(timespec='seconds'),
        "adstock_decay": best_decay,
        "spend_cols": spend_cols,
        "train_period": [str(data['date'].min().date()), str(data['date'].max().date())],
        "cleaning_thresholds": ingestor.cleaning_thresholds,
        "validation": {
            "method": f"expanding-window CV, {N_FOLDS} folds, tuned with linear model",
            "cv_mape_by_decay": cv_mape,
            "chosen_decay": best_decay,
        },
        "holdout": {},
        "channel_contributions": {},
        "channel_roi": {},
    }

    for name, model_cls in [("linear", LinearMMMModel), ("bayesian", BayesianMMMModel)]:
        print(f"\n=== {name} model ===")
        print("Test evaluation (untouched window)...")
        metrics = test_metrics(model_cls, pool_df, test_df, spend_cols, best_decay)
        summary["holdout"][name] = {k: round(v, 3) for k, v in metrics.items()}
        print(f"Out-of-sample: MAPE={metrics['mape']:.1f}%  R²={metrics['r2']:.3f}")

        print("Refitting on full data...")
        model = model_cls(adstock_decay=best_decay)
        model.train(data, 'sales', spend_cols)

        summary["channel_contributions"][name] = {
            ch: round(v, 0) for ch, v in model.channel_contributions(data).items()}
        summary["channel_roi"][name] = {
            ch: (round(v, 2) if v is not None else None)
            for ch, v in model.channel_roi(data).items()}

        if name == "linear":
            model.save(str(models_dir / "trained_linear_model.pkl"))
        else:
            model.save(str(models_dir / "trained_bayesian_trace.nc"))
        print(f"{name} model saved")

    summary_path = models_dir / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nTraining summary written to {summary_path}")
    print(json.dumps({k: summary[k] for k in ("validation", "holdout", "channel_roi")}, indent=2))


if __name__ == "__main__":
    train_models()
