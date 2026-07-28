#!/usr/bin/env python3
"""Train both MMM models with an honest temporal holdout evaluation.

Flow: ingest -> split (first 80% train / last 20% test) -> fit -> report
out-of-sample metrics -> refit on all data -> save artifacts + summary.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.linear_model import LinearMMMModel
from models.bayesian_mmm import BayesianMMMModel
from data.ingestion import DataIngestor
from evaluation.metrics import evaluate_model
from config import config


def holdout_metrics(model_cls, train_df, test_df, spend_cols):
    #fit on the past, evaluate on the future the model has never seen
    model = model_cls(adstock_decay=config.adstock_decay)
    model.train(train_df, 'sales', spend_cols)
    predictions = model.predict(test_df)
    return evaluate_model(test_df['sales'], predictions)


def train_models():
    print("Loading data...")
    ingestor = DataIngestor()
    data = ingestor.run(training=True)

    spend_cols = [col for col in data.columns if col.endswith('_spend')]
    data = data[['date', 'sales'] + spend_cols].copy()

    split = int(len(data) * 0.8)
    train_df, test_df = data.iloc[:split], data.iloc[split:]
    print(f"\nData: {data.shape}, channels: {spend_cols}")
    print(f"Holdout split: train {train_df['date'].min().date()}..{train_df['date'].max().date()}, "
          f"test {test_df['date'].min().date()}..{test_df['date'].max().date()}")
    print(f"adstock_decay: {config.adstock_decay}")

    models_dir = config.saved_models_path
    models_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "trained_at": datetime.now().isoformat(timespec='seconds'),
        "adstock_decay": config.adstock_decay,
        "spend_cols": spend_cols,
        "train_period": [str(data['date'].min().date()), str(data['date'].max().date())],
        "cleaning_thresholds": ingestor.cleaning_thresholds,
        "holdout": {},
        "channel_contributions": {},
        "channel_roi": {},
    }

    for name, model_cls in [("linear", LinearMMMModel), ("bayesian", BayesianMMMModel)]:
        print(f"\n=== {name} model ===")
        print("Holdout evaluation...")
        metrics = holdout_metrics(model_cls, train_df, test_df, spend_cols)
        summary["holdout"][name] = {k: round(v, 3) for k, v in metrics.items()}
        print(f"Out-of-sample: MAPE={metrics['mape']:.1f}%  R²={metrics['r2']:.3f}")

        print("Refitting on full data...")
        model = model_cls(adstock_decay=config.adstock_decay)
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
    print(json.dumps({k: summary[k] for k in ("holdout", "channel_roi")}, indent=2))


if __name__ == "__main__":
    train_models()
