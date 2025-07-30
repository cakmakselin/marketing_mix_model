import pytest
import json
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from datetime import date

with patch('api.main.MMMService') as mock_service_class:
    mock_service = Mock()
    mock_service.model.is_trained = True
    mock_service.model.adstock_decay = 0.3
    mock_service.evaluate.return_value = {'mape': 15.5}
    mock_service_class.return_value = mock_service
    
    from api.main import app

client = TestClient(app)

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
    def test_prediction_valid_request(self):
        # Reset the mock for this test
        from api.main import mmm_service
        mmm_service.model.is_trained = True
        mmm_service.model.predict = Mock(return_value=[1500.0, 1600.0])
        mmm_service.model.adstock_decay = 0.3
        
        request_data = {
            "data": [
                {
                    "date": "2024-01-01",
                    "tv_spend": 1000.0,
                    "social_media_spend": 500.0,
                    "search_spend": 300.0,
                    "radio_spend": 0.0,
                    "outdoor_spend": 0.0,
                    "print_spend": 0.0
                },
                {
                    "date": "2024-01-02",
                    "tv_spend": 1200.0,
                    "social_media_spend": 600.0,
                    "search_spend": 350.0,
                    "radio_spend": 200.0,
                    "outdoor_spend": 0.0,
                    "print_spend": 0.0
                }
            ]
        }
        
        response = client.post("/predictions", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "model_type" in data
        assert "adstock_decay" in data

    
    def test_prediction_service_not_trained(self):
        from api.main import mmm_service
        mmm_service.model.is_trained = False
        
        request_data = {
            "data": [
                {
                    "date": "2024-01-01",
                    "tv_spend": 1000.0,
                    "social_media_spend": 500.0,
                    "search_spend": 300.0,
                    "radio_spend": 0.0,
                    "outdoor_spend": 0.0,
                    "print_spend": 0.0
                }
            ]
        }
        
        response = client.post("/predictions", json=request_data)
        
        assert response.status_code == 400
        assert "Model not trained" in response.json()["detail"]
    
    def test_prediction_invalid_data(self):
        # Missing required date field
        invalid_request = {
            "data": [
                {
                    "tv_spend": 1000.0,
                    "social_media_spend": 500.0
                    # Missing date and other fields
                }
            ]
        }
        
        response = client.post("/predictions", json=invalid_request)
        
        assert response.status_code == 422  
    
    def test_prediction_empty_data(self):
        from api.main import mmm_service
        mmm_service.model.is_trained = True
        mmm_service.model.predict = Mock(return_value=[])
        
        request_data = {"data": []}
        
        response = client.post("/predictions", json=request_data)
        
        assert response.status_code == 200
    
    def test_prediction_default_values(self):
        from api.main import mmm_service
        mmm_service.model.is_trained = True
        mmm_service.model.predict = Mock(return_value=[1500.0])
        mmm_service.model.adstock_decay = 0.2
        
        request_data = {
            "data": [
                {
                    "date": "2024-01-01"
                    # All spend fields should default to 0.0
                }
            ]
        }
        
        response = client.post("/predictions", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 1

class TestModelEndpoints:
    def test_model_performance(self):
        from api.main import mmm_service
        mmm_service.evaluate = Mock(return_value={'mape': 12.5})
        mmm_service.model.adstock_decay = 0.4
        
        response = client.get("/models/performance")
        
        assert response.status_code == 200
        data = response.json()
        assert "performance" in data
        assert "model_type" in data
        assert "adstock_decay" in data
        assert data["performance"]["mape"] == 12.5
    
    def test_model_performance_error(self):
        from api.main import mmm_service
        mmm_service.evaluate = Mock(side_effect=Exception("Evaluation failed"))
        
        response = client.get("/models/performance")
        
        assert response.status_code == 400
        assert "Error" in response.json()["detail"]
    
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
    def test_spend_data_row_validation(self):
        from api.main import SpendDataRow
        
        # Valid data
        valid_data = {
            "date": date(2024, 1, 1),
            "tv_spend": 1000.0,
            "social_media_spend": 500.0
        }
        spend_row = SpendDataRow(**valid_data)
        assert spend_row.date == date(2024, 1, 1)
        assert spend_row.tv_spend == 1000.0
        assert spend_row.social_media_spend == 500.0
        # Other fields should default to 0.0
        assert spend_row.radio_spend == 0.0
    
    def test_prediction_request_validation(self):
        from api.main import PredictionRequest, SpendDataRow
        
        data = [
            SpendDataRow(
                date=date(2024, 1, 1),
                tv_spend=1000.0
            )
        ]
        
        request = PredictionRequest(data=data)
        assert len(request.data) == 1
        assert request.data[0].tv_spend == 1000.0

class TestAPIIntegration:
    def test_full_prediction_cycle(self):
        """Test the complete API request/response cycle"""
        from api.main import mmm_service
        mmm_service.model.is_trained = True
        mmm_service.model.predict = Mock(return_value=[1750.5, 1650.2])
        mmm_service.model.adstock_decay = 0.25
        
        request_data = {
            "data": [
                {
                    "date": "2024-01-01",
                    "tv_spend": 2000.0,
                    "social_media_spend": 800.0,
                    "search_spend": 500.0,
                    "radio_spend": 300.0,
                    "outdoor_spend": 1000.0,
                    "print_spend": 200.0
                },
                {
                    "date": "2024-01-02",
                    "tv_spend": 1500.0,
                    "social_media_spend": 600.0,
                    "search_spend": 400.0,
                    "radio_spend": 0.0,
                    "outdoor_spend": 800.0,
                    "print_spend": 100.0
                }
            ]
        }
        
        response = client.post("/predictions", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "predictions" in data
        assert "model_type" in data
        assert "adstock_decay" in data
        
        assert isinstance(data["predictions"], list)
        assert len(data["predictions"]) == 2
        assert all(isinstance(pred, (int, float)) for pred in data["predictions"])
        assert data["adstock_decay"] == 0.25
        
        mmm_service.model.predict.assert_called_once() 