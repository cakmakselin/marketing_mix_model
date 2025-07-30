import pytest
import json
import io
import tempfile
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from datetime import date
import pandas as pd

with patch('api.main.MMMService') as mock_service_class:
    mock_service = Mock()
    mock_service.model.is_trained = True
    mock_service.model.adstock_decay = 0.3
    mock_service.evaluate.return_value = {'mape': 15.5}
    mock_service_class.return_value = mock_service
    
    from api.main import app

client = TestClient(app)

def create_test_csv_files():
    """Helper function to create test CSV files in memory"""
    # Create TV spend CSV
    tv_data = pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02'],
        'tv_spend': [1000.0, 1200.0]
    })
    tv_csv = io.StringIO()
    tv_data.to_csv(tv_csv, index=False)
    tv_csv.seek(0)
    
    # Create social media spend CSV
    social_data = pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02'],
        'social_media_spend': [500.0, 600.0]
    })
    social_csv = io.StringIO()
    social_data.to_csv(social_csv, index=False)
    social_csv.seek(0)
    
    # Create search spend CSV
    search_data = pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02'],
        'search_spend': [300.0, 350.0]
    })
    search_csv = io.StringIO()
    search_data.to_csv(search_csv, index=False)
    search_csv.seek(0)
    
    return {
        'tv_spend.csv': tv_csv.getvalue(),
        'social_media_spend.csv': social_csv.getvalue(),
        'search_spend.csv': search_csv.getvalue()
    }

class TestHealthEndpoint:
    def test_health_check_success(self):
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "service_ready" in data
        assert "model_type" in data
        assert "adstock_decay" in data
    
    def test_health_check_content(self):
        response = client.get("/health")
        data = response.json()
        
        assert isinstance(data["service_ready"], bool)
        assert isinstance(data["adstock_decay"], (int, float))

class TestPredictionEndpoint:
    @patch('api.main.DataIngestor')
    def test_prediction_valid_request(self, mock_ingestor_class):
        # Setup mock DataIngestor
        mock_ingestor = Mock()
        mock_data = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01', '2024-01-02']),
            'tv_spend': [1000.0, 1200.0],
            'social_media_spend': [500.0, 600.0],
            'search_spend': [300.0, 350.0]
        })
        mock_ingestor.run.return_value = mock_data
        mock_ingestor_class.return_value = mock_ingestor
        
        # Setup model mock
        from api.main import mmm_service
        mmm_service.model.is_trained = True
        mmm_service.model.predict = Mock(return_value=[1500.0, 1600.0])
        mmm_service.model.adstock_decay = 0.3
        
        # Create test files
        test_files = create_test_csv_files()
        files = [
            ("files", ("tv_spend.csv", io.BytesIO(test_files['tv_spend.csv'].encode()), "text/csv")),
            ("files", ("social_media_spend.csv", io.BytesIO(test_files['social_media_spend.csv'].encode()), "text/csv")),
            ("files", ("search_spend.csv", io.BytesIO(test_files['search_spend.csv'].encode()), "text/csv"))
        ]
        
        response = client.post("/predictions", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert "forecast" in data
        assert "model_type" in data
        assert "adstock_decay" in data
        assert "rows_processed" in data
        assert len(data["forecast"]) == 2

    @patch('api.main.DataIngestor')
    def test_prediction_service_not_trained(self, mock_ingestor_class):
        # Setup mock DataIngestor
        mock_ingestor = Mock()
        mock_data = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01']),
            'tv_spend': [1000.0]
        })
        mock_ingestor.run.return_value = mock_data
        mock_ingestor_class.return_value = mock_ingestor
        
        from api.main import mmm_service
        mmm_service.model.is_trained = False
        
        # Create test files
        test_files = create_test_csv_files()
        files = [
            ("files", ("tv_spend.csv", io.BytesIO(test_files['tv_spend.csv'].encode()), "text/csv"))
        ]
        
        response = client.post("/predictions", files=files)
        
        assert response.status_code == 400
        assert "Model not trained" in response.json()["detail"]
    
    def test_prediction_invalid_data(self):
        # Test without any files
        response = client.post("/predictions", files=[])
        
        assert response.status_code == 422  
    
    @patch('api.main.DataIngestor')
    def test_prediction_empty_data(self, mock_ingestor_class):
        # Setup mock DataIngestor with empty data
        mock_ingestor = Mock()
        mock_data = pd.DataFrame(columns=['date'])
        mock_ingestor.run.return_value = mock_data
        mock_ingestor_class.return_value = mock_ingestor
        
        from api.main import mmm_service
        mmm_service.model.is_trained = True
        mmm_service.model.predict = Mock(return_value=[])
        
        # Create minimal test file
        empty_csv = "date\n"
        files = [
            ("files", ("empty.csv", io.BytesIO(empty_csv.encode()), "text/csv"))
        ]
        
        response = client.post("/predictions", files=files)
        
        assert response.status_code == 400
    
    @patch('api.main.DataIngestor')
    def test_prediction_default_values(self, mock_ingestor_class):
        # Setup mock DataIngestor
        mock_ingestor = Mock()
        mock_data = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01']),
            'tv_spend': [0.0]  # Default value
        })
        mock_ingestor.run.return_value = mock_data
        mock_ingestor_class.return_value = mock_ingestor
        
        from api.main import mmm_service
        mmm_service.model.is_trained = True
        mmm_service.model.predict = Mock(return_value=[1500.0])
        mmm_service.model.adstock_decay = 0.2
        
        # Create test file with minimal data
        minimal_csv = "date,tv_spend\n2024-01-01,0.0\n"
        files = [
            ("files", ("minimal.csv", io.BytesIO(minimal_csv.encode()), "text/csv"))
        ]
        
        response = client.post("/predictions", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["forecast"]) == 1

class TestModelEndpoints:
    @pytest.mark.skip(reason="Performance endpoint not implemented")
    def test_model_performance(self):
        pass
    
    @pytest.mark.skip(reason="Performance endpoint not implemented")
    def test_model_performance_error(self):
        pass
    
    def test_model_info(self):
        from api.main import mmm_service
        mmm_service.model.adstock_decay = 0.35
        mmm_service.model.is_trained = True
        
        response = client.get("/models")
        
        assert response.status_code == 200
        data = response.json()
        assert "model_type" in data
        assert "adstock_decay" in data
        assert "is_trained" in data
        assert data["adstock_decay"] == 0.35
        assert data["is_trained"] is True

class TestAPIValidation:
    @pytest.mark.skip(reason="SpendDataRow not available in current API")
    def test_spend_data_row_validation(self):
        pass
    
    @pytest.mark.skip(reason="PredictionRequest not available in current API")
    def test_prediction_request_validation(self):
        pass

class TestAPIIntegration:
    @patch('api.main.DataIngestor')
    def test_full_prediction_cycle(self, mock_ingestor_class):
        """Test the complete API request/response cycle"""
        # Setup mock DataIngestor
        mock_ingestor = Mock()
        mock_data = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01', '2024-01-02']),
            'tv_spend': [2000.0, 1500.0],
            'social_media_spend': [800.0, 600.0],
            'search_spend': [500.0, 400.0],
            'radio_spend': [300.0, 0.0],
            'outdoor_spend': [1000.0, 800.0],
            'print_spend': [200.0, 100.0]
        })
        mock_ingestor.run.return_value = mock_data
        mock_ingestor_class.return_value = mock_ingestor
        
        from api.main import mmm_service
        mmm_service.model.is_trained = True
        mmm_service.model.predict = Mock(return_value=[1750.5, 1650.2])
        mmm_service.model.adstock_decay = 0.25
        
        # Create comprehensive test files
        test_files = create_test_csv_files()
        files = [
            ("files", ("tv_spend.csv", io.BytesIO(test_files['tv_spend.csv'].encode()), "text/csv")),
            ("files", ("social_media_spend.csv", io.BytesIO(test_files['social_media_spend.csv'].encode()), "text/csv")),
            ("files", ("search_spend.csv", io.BytesIO(test_files['search_spend.csv'].encode()), "text/csv"))
        ]
        
        response = client.post("/predictions", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "forecast" in data
        assert "model_type" in data
        assert "adstock_decay" in data
        assert "rows_processed" in data
        
        assert isinstance(data["forecast"], list)
        assert len(data["forecast"]) == 2
        assert all("date" in item and "predicted_sales" in item for item in data["forecast"])
        assert data["adstock_decay"] == 0.25
        
        mmm_service.model.predict.assert_called_once() 