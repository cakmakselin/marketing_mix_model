from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


class MMMConfig:
    raw_data_path = resolve_path(os.getenv("MMM_RAW_DATA_PATH", "data_files/raw/"))
    processed_data_path = resolve_path(os.getenv("MMM_PROCESSED_DATA_PATH", "data_files/processed/"))
    saved_models_path = resolve_path(os.getenv("MMM_SAVED_MODELS_PATH", "models/saved_models/"))

    spend_file_pattern = "*_spend*"
    sales_file_name = "sales_data"

    default_model_type = "bayesian"
    adstock_decay = 0.3


config = MMMConfig() 