import pandas as pd
import pickle
from pathlib import Path
from config import config


def save_processed_data(df: pd.DataFrame, output_dir: Path = None):
    output_dir = output_dir or config.processed_data_path
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = "cleaned_data"
    parquet_path = output_dir / f"{filename}.parquet"
    pickle_path = output_dir / f"{filename}.pkl"

    try:
        df.to_parquet(parquet_path, engine='pyarrow', compression='snappy')
        print(f"Data saved to {parquet_path}")
    except ImportError:
        with open(pickle_path, 'wb') as f:
            pickle.dump(df, f)
        print(f"Data saved to {pickle_path}")


def load_processed_data(input_dir: Path = None) -> pd.DataFrame:
    input_dir = input_dir or config.processed_data_path

    filename = "cleaned_data"
    parquet_path = input_dir / f"{filename}.parquet"
    pickle_path = input_dir / f"{filename}.pkl"

    if parquet_path.exists():
        try:
            df = pd.read_parquet(parquet_path, engine='pyarrow')
            print(f"Data loaded from {parquet_path}")
            return df
        except ImportError:
            pass

    if pickle_path.exists():
        with open(pickle_path, 'rb') as f:
            df = pickle.load(f)
        print(f"Data loaded from {pickle_path}")
        return df

    raise FileNotFoundError(f"No processed data found at {parquet_path} or {pickle_path}")


def processed_data_exists(input_dir: Path = None) -> bool:
    input_dir = input_dir or config.processed_data_path

    filename = "cleaned_data"
    parquet_path = input_dir / f"{filename}.parquet"
    pickle_path = input_dir / f"{filename}.pkl"

    return parquet_path.exists() or pickle_path.exists()
