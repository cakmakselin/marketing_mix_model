import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from data.ingestion import DataIngestor
from data.validation import SpendDataValidator, SalesDataValidator, MergedDataValidator, CleanedDataValidator
from data.storage import save_processed_data, load_processed_data, processed_data_exists

class TestDataIngestor:
    def test_initialization(self):
        ingestor = DataIngestor()
        assert ingestor.spend_df is None
        assert ingestor.sales_df is None
        assert ingestor.merged_df is None
        assert ingestor.cleaned_df is None
    
    def test_load_spend_data(self, sample_spend_files):
        ingestor = DataIngestor(data_dir=sample_spend_files)
        ingestor.load_spend_data()
        assert ingestor.spend_df is not None
        assert 'date' in ingestor.spend_df.columns
        assert len(ingestor.spend_df.columns) > 2
    
    def test_load_sales_data(self, sample_sales_file):
        ingestor = DataIngestor(data_dir=sample_sales_file.parent)
        ingestor.load_sales_data()
        assert ingestor.sales_df is not None
        assert 'sales' in ingestor.sales_df.columns
    
    def test_full_pipeline(self, sample_data_dir):
        ingestor = DataIngestor(data_dir=sample_data_dir)
        result = ingestor.run()
        assert result is not None
        assert ingestor.cleaned_df is not None
        assert len(ingestor.cleaned_df) > 0
        # training run must fit and expose cleaning thresholds for serving
        assert ingestor.cleaning_thresholds
        assert 'sales' in ingestor.cleaning_thresholds

    def test_prediction_mode_without_sales(self, sample_spend_files):
        # prediction uploads have no sales file and can be short
        ingestor = DataIngestor(data_dir=sample_spend_files)
        result = ingestor.run(training=False)
        assert 'sales' not in result.columns
        assert len(result) == 10

    def test_training_mode_requires_sales(self, sample_spend_files):
        ingestor = DataIngestor(data_dir=sample_spend_files)
        with pytest.raises(FileNotFoundError, match="Sales file not found"):
            ingestor.run(training=True)

    def test_prediction_mode_reuses_thresholds(self, sample_spend_files):
        # a value below the training threshold must survive serving-time
        # cleaning even if it is extreme relative to the uploaded window
        thresholds = {'tv_spend': 10_000_000, 'radio_spend': 10_000_000}
        ingestor = DataIngestor(data_dir=sample_spend_files)
        result = ingestor.run(training=False, cleaning_thresholds=thresholds)
        assert ingestor.cleaning_thresholds == thresholds
        assert result[['tv_spend', 'radio_spend']].notna().all().all()

    def test_leading_outlier_gets_filled(self, tmp_path):
        # a flagged first row used to stay NaN and crash validation
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        n = 35
        tv = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=n),
            'spend': [-50.0] + [100.0] * (n - 1),  # negative on day one
        })
        tv.to_csv(raw_dir / "tv_spend.csv", index=False)
        radio = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=n),
            'spend': [200.0] * n,
        })
        radio.to_csv(raw_dir / "radio_spend.csv", index=False)
        sales = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=n),
            'sales': [1000.0] * n,
        })
        sales.to_csv(raw_dir / "sales_data.csv", index=False)

        result = DataIngestor(data_dir=raw_dir).run()
        assert result['tv_spend'].notna().all()
        assert result['tv_spend'].iloc[0] == 100.0

class TestValidators:
    def test_spend_validator_valid(self):
        df = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02'],
            'spend': [100, 200]
        })
        # Should not raise exception
        SpendDataValidator(date=df['date'], spend_value=df['spend'])
    
    def test_spend_validator_invalid(self):
        df = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02'],
            'spend': ['not_numeric', 'also_not_numeric']
        })
        with pytest.raises(ValueError):
            SpendDataValidator(date=df['date'], spend_value=df['spend'])
    
    def test_sales_validator_valid(self):
        df = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02'],
            'sales': [1000, 2000]
        })
        # Should not raise exception
        SalesDataValidator(date=df['date'], sales_value=df['sales'])
    
    def test_cleaned_validator_catches_negatives(self):
        spend_cols = {'tv_spend': pd.Series([-100, 200, 300])}
        sales = pd.Series([1000, 2000, 3000])
        date = pd.Series(pd.date_range('2023-01-01', periods=3))
        
        with pytest.raises(ValueError, match="still has negative values"):
            CleanedDataValidator(date=date, spend_columns=spend_cols, sales=sales)

class TestStorage:
    def test_save_and_load(self, tmp_path):
        # Mock config for testing
        import data.storage as storage_module
        original_config = storage_module.config
        
        class MockConfig:
            processed_data_path = tmp_path
        
        storage_module.config = MockConfig()
        
        try:
            test_df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
            
            save_processed_data(test_df)
            loaded_df = load_processed_data()
            
            pd.testing.assert_frame_equal(test_df, loaded_df)
        finally:
            storage_module.config = original_config
    
    def test_data_exists(self, tmp_path):
        import data.storage as storage_module
        original_config = storage_module.config
        
        class MockConfig:
            processed_data_path = tmp_path
        
        storage_module.config = MockConfig()
        
        try:
            assert not processed_data_exists()
            
            test_df = pd.DataFrame({'col1': [1, 2]})
            save_processed_data(test_df)
            
            assert processed_data_exists()
        finally:
            storage_module.config = original_config

# Fixtures
@pytest.fixture
def sample_spend_files(tmp_path):
    spend_dir = tmp_path / "raw"
    spend_dir.mkdir()
    
    # Create spend files
    for channel in ['radio', 'tv']:
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'spend': np.random.randint(100, 1000, 10)
        })
        df.to_csv(spend_dir / f"{channel}_spend.csv", index=False)
    
    return spend_dir

@pytest.fixture
def sample_sales_file(tmp_path):
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    
    df = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=10),
        'sales': np.random.randint(1000, 5000, 10)
    })
    df.to_csv(data_dir / "sales_data.csv", index=False)
    
    return data_dir / "sales_data.csv"

@pytest.fixture
def sample_data_dir(tmp_path):
    """Create sample spend and sales files"""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    
    # Create spend files with 35 days of data
    tv_data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=35),
        'tv_spend': np.random.normal(500, 50, 35)
    })
    tv_data.to_csv(raw_dir / "tv_spend.csv", index=False)
    
    radio_data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=35),
        'radio_spend': np.random.normal(200, 20, 35)
    })
    radio_data.to_csv(raw_dir / "radio_spend.csv", index=False)
    
    # Create sales file with 35 days of data
    sales_data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=35),
        'sales': np.random.normal(1000, 100, 35)
    })
    sales_data.to_csv(raw_dir / "sales_data.csv", index=False)
    
    return raw_dir 