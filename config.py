from pathlib import Path

class MMMConfig:
    #data paths
    raw_data_path = Path("data_files/raw/")
    processed_data_path = Path("data_files/processed/")
    
    #file patterns
    spend_file_pattern = "*_spend*"
    sales_file_name = "sales_data"
    
    #model settings
    default_model_type = "bayesian"
    adstock_decay = 0.3
    
config = MMMConfig() 