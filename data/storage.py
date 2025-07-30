import pandas as pd
from pathlib import Path
from config import config

def save_processed_data(df: pd.DataFrame, output_dir: Path = None):
    #save cleaned data to parquet
    output_dir = output_dir or config.processed_data_path
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = "cleaned_data"
    filepath = output_dir / f"{filename}.parquet"
    
    df.to_parquet(filepath, engine='pyarrow', compression='snappy')
    print(f"Data saved to {filepath}")

def load_processed_data(input_dir: Path = None) -> pd.DataFrame:
    #load cleaned data from parquet
    input_dir = input_dir or config.processed_data_path
    
    filename = "cleaned_data"
    filepath = input_dir / f"{filename}.parquet"
    
    if not filepath.exists():
        raise FileNotFoundError(f"No processed data found at {filepath}")
    
    df = pd.read_parquet(filepath, engine='pyarrow')
    print(f"Data loaded from {filepath}")
    return df

def processed_data_exists(input_dir: Path = None) -> bool:
    #check if processed data file exists
    input_dir = input_dir or config.processed_data_path
    
    filename = "cleaned_data"
    filepath = input_dir / f"{filename}.parquet"
    
    return filepath.exists()
