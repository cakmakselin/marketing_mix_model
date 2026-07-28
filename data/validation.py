from pydantic import BaseModel, field_validator, ConfigDict, model_validator
from typing import Dict, Optional
import pandas as pd

class SpendDataValidator(BaseModel):
    date: pd.Series
    spend_value: pd.Series
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @field_validator('date')
    @classmethod
    def validate_date(cls, v):
        try:
            pd.to_datetime(v)
            return v
        except (ValueError, TypeError) as e:
            raise ValueError(f"Date column must be convertible to datetime: {e}")
    
    @field_validator('spend_value')
    @classmethod
    def validate_spend(cls, v):
        if not pd.api.types.is_numeric_dtype(v):
            raise ValueError("Spend values must be numeric")
        return v

class SalesDataValidator(BaseModel):
    date: pd.Series
    sales_value: pd.Series
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @field_validator('date')
    @classmethod
    def validate_date(cls, v):
        try:
            pd.to_datetime(v)
            return v
        except (ValueError, TypeError) as e:
            raise ValueError(f"Date column must be convertible to datetime: {e}")
    
    @field_validator('sales_value')
    @classmethod
    def validate_sales(cls, v):
        if not pd.api.types.is_numeric_dtype(v):
            raise ValueError("Sales values must be numeric")
        return v

class MergedDataValidator(BaseModel):
    date: pd.Series
    spend_columns: Dict[str, pd.Series]
    #sales is optional: prediction-only data has no sales column
    sales: Optional[pd.Series] = None
    #training needs enough history; prediction can be any length
    min_days: int = 30

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode='after')
    def validate_date_range(self):
        if len(self.date) < self.min_days:
            raise ValueError(f"Need at least {self.min_days} days of data")
        return self

    @field_validator('spend_columns')
    def validate_spend_channels(cls, v):
        #need multiple channels for MMM
        if len(v) < 2:
            raise ValueError(f"Need at least 2 spend channels")
        return v

    @field_validator('sales')
    @classmethod
    def validate_sales(cls, v):
        if v is not None and not pd.api.types.is_numeric_dtype(v):
            raise ValueError("Sales must be numeric")
        return v

class CleanedDataValidator(BaseModel):
    date: pd.Series
    spend_columns: Dict[str, pd.Series]
    sales: Optional[pd.Series] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @field_validator('spend_columns')
    @classmethod
    def validate_cleaned_spend(cls, v):
        for name, series in v.items():
            # Check cleaning worked
            if (series < 0).any():
                raise ValueError(f"{name} still has negative values")
            if series.isnull().any():
                raise ValueError(f"{name} still has missing values")
        return v
    
    @field_validator('sales')
    @classmethod
    def validate_cleaned_sales(cls, v):
        if v is None:
            return v
        if (v < 0).any():
            raise ValueError("Sales still has negative values")
        if v.isnull().any():
            raise ValueError("Sales still has missing values")
        if (v == 0).sum() > 0:
            print(f"Warning: {(v == 0).sum()} zero sales values remain")
        return v 