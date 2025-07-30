import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from models.linear_model import LinearMMMModel
from models.bayesian_mmm import BayesianMMMModel
from models.base_model import BaseMMMModel

class TestBaseMMMModel:
    def test_initialization(self):
        model = BaseMMMModel(adstock_decay=0.3)
        assert model.adstock_decay == 0.3
        assert not model.is_trained
    
    def test_apply_adstock_basic(self):
        model = BaseMMMModel(adstock_decay=0.5)
        spend_series = pd.Series([100, 0, 0, 0])
        result = model.apply_adstock(spend_series, 0.5)
        
        # Check carryover effect: 100 -> 50 -> 25 -> 12.5
        assert result[0] == 100
        assert result[1] == 50
        assert result[2] == 25
        assert result[3] == 12.5
    
    def test_apply_adstock_no_decay(self):
        model = BaseMMMModel(adstock_decay=0.0)
        spend_series = pd.Series([100, 0, 0])
        result = model.apply_adstock(spend_series, 0.0)
        
        # No carryover with decay=0
        assert result[0] == 100
        assert result[1] == 0
        assert result[2] == 0
    
    def test_apply_log_transform(self):
        model = BaseMMMModel()
        series = pd.Series([0, 100, 1000])
        result = model.apply_log_transform(series)
        
        # log1p: log(1+0)=0, log(1+100)≈4.6, log(1+1000)≈6.9
        assert result[0] == 0
        assert result[1] == pytest.approx(4.6, rel=0.1)
        assert result[2] == pytest.approx(6.9, rel=0.1)
    
    def test_add_features(self):
        model = BaseMMMModel(adstock_decay=0.2)
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=3),
            'sales': [1000, 1100, 1200],
            'tv_spend': [500, 0, 300],
            'radio_spend': [200, 100, 0]
        })
        
        result = model.add_features(df, ['tv_spend', 'radio_spend'])
        
        # Should have original + adstock + log + adstock_log for each spend column
        expected_cols = [
            'date', 'sales', 'tv_spend', 'radio_spend',
            'tv_spend_adstock', 'tv_spend_log', 'tv_spend_adstock_log',
            'radio_spend_adstock', 'radio_spend_log', 'radio_spend_adstock_log',
            'sales_log'
        ]
        
        for col in expected_cols:
            assert col in result.columns

class TestLinearMMMModel:
    def test_initialization(self):
        model = LinearMMMModel(adstock_decay=0.4)
        assert model.adstock_decay == 0.4
        assert not model.is_trained
        assert model.model is not None
    
    def test_train_basic(self, sample_training_data):
        model = LinearMMMModel(adstock_decay=0.3)
        df, spend_cols = sample_training_data
        
        model.train(df, 'sales', spend_cols)
        
        assert model.is_trained
        assert model.feature_cols is not None
        assert len(model.feature_cols) > 0
    
    def test_predict_without_training(self, sample_training_data):
        model = LinearMMMModel()
        df, _ = sample_training_data
        
        with pytest.raises(AttributeError):
            model.predict(df)
    
    def test_predict_after_training(self, sample_training_data):
        model = LinearMMMModel(adstock_decay=0.2)
        df, spend_cols = sample_training_data
        
        model.train(df, 'sales', spend_cols)
        predictions = model.predict(df)
        
        assert len(predictions) == len(df)
        assert all(pred > 0 for pred in predictions)  # Sales should be positive

class TestBayesianMMMModel:
    def test_initialization(self):
        model = BayesianMMMModel(adstock_decay=0.5)
        assert model.adstock_decay == 0.5
        assert not model.is_trained
        assert model.trace is None
    
    @patch('pymc.sample') 
    def test_train_basic(self, mock_sample, sample_training_data):
        mock_trace = Mock()
        mock_trace.posterior = {
            'alpha': Mock(mean=Mock(return_value=Mock(values=1000))),
            'betas': Mock(mean=Mock(return_value=Mock(values=np.array([1, 2, 3, 4]))))
        }
        mock_sample.return_value = mock_trace
        
        model = BayesianMMMModel(adstock_decay=0.3)
        df, spend_cols = sample_training_data
        
        model.train(df, 'sales', spend_cols, draws=50) 
        
        assert model.is_trained
        assert model.alpha_mean is not None
        assert model.betas_mean is not None
    
    def test_predict_without_training(self, sample_training_data):
        model = BayesianMMMModel()
        df, _ = sample_training_data
        
        with pytest.raises(AttributeError):
            model.predict(df)
    
    @patch('pymc.sample')
    def test_save_and_load_trace(self, mock_sample, tmp_path):
        mock_trace = Mock()
        mock_trace.posterior = {
            'alpha': Mock(mean=Mock(return_value=Mock(values=1000))),
            'betas': Mock(mean=Mock(return_value=Mock(values=np.array([1, 2]))))
        }
        mock_trace.to_netcdf = Mock()
        mock_sample.return_value = mock_trace
        
        model = BayesianMMMModel()
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5),
            'sales': [1000, 1100, 1200, 1300, 1400],
            'tv_spend': [500, 600, 700, 800, 900]
        })
        
        model.train(df, 'sales', ['tv_spend'], draws=50)
        
        filepath = tmp_path / "test_trace.nc"
        model.save_trace(str(filepath))
        
        mock_trace.to_netcdf.assert_called_once_with(str(filepath))

# Fixtures
@pytest.fixture
def sample_training_data():
    """Create sample data for model training"""
    np.random.seed(42)  
    
    df = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=30),
        'sales': np.random.normal(1000, 100, 30),
        'tv_spend': np.random.normal(500, 50, 30),
        'radio_spend': np.random.normal(200, 20, 30),
        'social_media_spend': np.random.normal(100, 10, 30)
    })
    
    for col in ['sales', 'tv_spend', 'radio_spend', 'social_media_spend']:
        df[col] = df[col].abs()
    
    spend_cols = ['tv_spend', 'radio_spend', 'social_media_spend']
    
    return df, spend_cols 