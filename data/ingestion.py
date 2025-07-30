import pandas as pd
from pathlib import Path
import numpy as np
from data.validation import SpendDataValidator, SalesDataValidator, MergedDataValidator, CleanedDataValidator
from data.storage import save_processed_data
from config import config

class DataIngestor:
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or config.raw_data_path
        self.spend_df = None
        self.sales_df = None
        self.merged_df = None
        self.cleaned_df = None

    def load_spend_data(self):
        spend_files = list(self.data_dir.glob(config.spend_file_pattern))
        if not spend_files:
            raise ValueError(f"No spend files found matching: {config.spend_file_pattern}")
        
        print(f"Loading {len(spend_files)} spend files...")
        spend_dfs = {}
        
        for file in spend_files:
            df = pd.read_csv(file)
            
            #validate
            if len(df.columns) != 2:
                raise ValueError(f"{file.stem} must have exactly 2 columns")
            SpendDataValidator(date=df.iloc[:, 0], spend_value=df.iloc[:, 1])
            
            #format
            df.columns = ["date", file.stem]
            df["date"] = pd.to_datetime(df["date"])
            spend_dfs[file.stem] = df

        #merge on date
        merged_df = list(spend_dfs.values())[0]
        for df in list(spend_dfs.values())[1:]:
            merged_df = pd.merge(merged_df, df, on="date", how='outer')
        self.spend_df = merged_df

    def load_sales_data(self):
        print("Loading sales data...")
        sales_file = self.data_dir / f"{config.sales_file_name}.csv"
        
        if not sales_file.exists():
            raise FileNotFoundError(f"Sales file not found: {sales_file}")
        
        self.sales_df = pd.read_csv(sales_file)
        
        #validate
        if len(self.sales_df.columns) != 2:
            raise ValueError("Sales file must have exactly 2 columns")
        SalesDataValidator(date=self.sales_df.iloc[:, 0], sales_value=self.sales_df.iloc[:, 1])
        
        #format
        self.sales_df.columns = ["date", "sales"]
        self.sales_df["date"] = pd.to_datetime(self.sales_df["date"])

    def merge_data(self):
        if self.spend_df is None or self.sales_df is None:
            raise ValueError("Load spend and sales data first")
        
        print("Merging data...")
        self.merged_df = pd.merge(self.spend_df, self.sales_df, on="date", how='left')
        
        #validate
        spend_cols = {col: self.merged_df[col] for col in self.merged_df.columns 
                     if col not in ["date", "sales"]}
        MergedDataValidator(
            date=self.merged_df["date"],
            spend_columns=spend_cols,
            sales=self.merged_df["sales"]
        )

    def clean_data(self):
        print("Cleaning data...")
        df_clean = self.merged_df.copy()

        for column in self.merged_df.select_dtypes(include=[np.number]).columns:    
            outliers_to_fix = pd.Series([False] * len(self.merged_df), index=self.merged_df.index)
            
            #outliers
            normal_high = self.merged_df[column].quantile(0.95)
            extreme_threshold = normal_high * 20
            extreme_outliers = self.merged_df[column] > extreme_threshold
            outliers_to_fix |= extreme_outliers
            print(f"  {column} - extreme highs (>{extreme_threshold:,.0f}): {extreme_outliers.sum()}")
            
            negative_outliers = self.merged_df[column] < 0
            outliers_to_fix |= negative_outliers
            print(f"  {column} - negative values: {negative_outliers.sum()}")
            
            zero_issues = pd.Series([False] * len(self.merged_df), index=self.merged_df.index)
            if column == "sales":
                zero_issues = self.merged_df[column] == 0
                outliers_to_fix |= zero_issues
                print(f"  {column} - zero sales days: {zero_issues.sum()}")
            
            existing_nan = self.merged_df[column].isna()
            outliers_to_fix |= existing_nan
            print(f"  {column} - missing values: {existing_nan.sum()}")
            
            #interpolation
            total_issues = outliers_to_fix.sum()
            if total_issues > 0:
                df_clean.loc[outliers_to_fix, column] = np.nan
                df_clean[column] = df_clean[column].interpolate(method='linear')

        self.cleaned_df = df_clean
        
        # Validate cleaning worked
        spend_cols = {col: self.cleaned_df[col] for col in self.cleaned_df.columns 
                     if col not in ["date", "sales"]}
        CleanedDataValidator(
            date=self.cleaned_df["date"],
            spend_columns=spend_cols,
            sales=self.cleaned_df["sales"]
        )

    def save_processed_data(self):
        if self.cleaned_df is None:
            raise ValueError("No cleaned data to save")
        save_processed_data(self.cleaned_df)
        print("Data saved successfully")

    def run(self):
        self.load_spend_data()
        self.load_sales_data()
        self.merge_data()
        self.clean_data()
        print("Data ingestion complete")
        return self.cleaned_df

if __name__ == "__main__":
    ingestor = DataIngestor()
    ingestor.run()