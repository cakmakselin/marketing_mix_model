#!/usr/bin/env python3
"""
Simple model training script that uses config values directly
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.linear_model import LinearMMMModel
from models.bayesian_mmm import BayesianMMMModel
from data.ingestion import DataIngestor
from config import config

def train_models():
    """Train both models using config parameters"""
    print("Loading data...")
    ingestor = DataIngestor()
    data = ingestor.run()
    
    # Filter to spend columns only
    spend_cols = [col for col in data.columns if col.endswith('_spend')]
    data = data[['date', 'sales'] + spend_cols].copy()
    
    print(f"Data shape: {data.shape}")
    print(f"Spend columns: {spend_cols}")
    print(f"Using adstock_decay: {config.adstock_decay}")
    
    models_dir = Path("models/saved_models")
    models_dir.mkdir(exist_ok=True)
    
    # Train Linear Model
    print("\nTraining Linear Model...")
    linear_model = LinearMMMModel(adstock_decay=config.adstock_decay)
    linear_model.train(data, 'sales', spend_cols)
    print("Linear model training complete")
    
    # Save Linear Model
    linear_model.save(str(models_dir / "trained_linear_model.pkl"))
    print("Linear model saved")
    
    # Train Bayesian Model  
    print("\nTraining Bayesian Model...")
    bayesian_model = BayesianMMMModel(adstock_decay=config.adstock_decay)
    bayesian_model.train(data, 'sales', spend_cols, draws=500)
    print("Bayesian model training complete")
    
    # Save Bayesian trace (Bayesian model only uses trace files)
    if bayesian_model.trace is not None:
        bayesian_model.save_trace(str(models_dir / "trained_bayesian_trace.nc"))
        print("Bayesian trace saved")
    else:
        print("Warning: No trace to save for Bayesian model")
    
    print(f"\nTraining complete using {config.default_model_type} as default model type")
    print(f"Models saved to: {models_dir}")

if __name__ == "__main__":
    train_models() 