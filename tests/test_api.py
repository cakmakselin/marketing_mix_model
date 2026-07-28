"""End-to-end API tests: no mocks.

These exercise the real service, real ingestion and the real trained
artifacts in models/saved_models/. They are the tests that catch broken
serving paths (missing features, stale artifacts, model-load failures).
"""
import io
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from api.main import app, mmm_service
from config import config

RAW = config.raw_data_path


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # runs lifespan -> loads pretrained model
        yield c


@pytest.fixture(scope="module")
def loaded(client):
    if not mmm_service.model.is_trained:
        pytest.skip("No loadable pretrained model in this environment "
                    "(run scripts/train_models.py first)")


def spend_files(channels=('tv', 'radio', 'social_media', 'search', 'outdoor', 'print')):
    return [("files", (f"{ch}_spend.csv", open(RAW / f"{ch}_spend.csv", "rb"), "text/csv"))
            for ch in channels]


class TestHealthEndpoint:
    def test_health_reports_readiness_honestly(self, client):
        data = client.get("/health").json()
        assert data["service_ready"] == mmm_service.model.is_trained
        assert data["status"] == ("ok" if data["service_ready"] else "degraded")


class TestPredictionEndpoint:
    def test_spend_only_upload(self, client, loaded):
        # core use case: predictions from spend files, no sales file
        response = client.post("/predictions", files=spend_files())
        assert response.status_code == 200

        data = response.json()
        assert data["rows_processed"] == 365
        assert data["evaluation"] is None
        assert len(data["forecast"]) == 365

        row = data["forecast"][0]
        assert set(row) >= {"date", "predicted_sales"}
        # forecasts must be in a sane range, not parroted actuals or zeros
        preds = [r["predicted_sales"] for r in data["forecast"]]
        assert 10_000 < np.mean(preds) < 1_000_000

    def test_upload_with_sales_gets_evaluation(self, client, loaded):
        files = spend_files() + [
            ("files", ("sales_data.csv", open(RAW / "sales_data.csv", "rb"), "text/csv"))]
        response = client.post("/predictions", files=files)
        assert response.status_code == 200

        evaluation = response.json()["evaluation"]
        assert "mape" in evaluation and "r2" in evaluation
        assert evaluation["mape"] < 50

    def test_bayesian_forecasts_include_intervals(self, client, loaded):
        if config.default_model_type != "bayesian" or mmm_service.model.trace is None:
            pytest.skip("intervals need the bayesian model with a loaded trace")
        response = client.post("/predictions", files=spend_files())
        row = response.json()["forecast"][0]
        assert row["lower_90"] <= row["predicted_sales"] <= row["upper_90"]

    def test_missing_channel_is_client_error(self, client, loaded):
        # model trained on 6 channels; uploading 2 must fail clearly, not misalign
        response = client.post("/predictions", files=spend_files(('tv', 'radio')))
        assert response.status_code == 400
        assert "Missing spend columns" in response.json()["detail"]

    def test_no_files_is_client_error(self, client):
        assert client.post("/predictions", files=[]).status_code == 422

    def test_not_trained_returns_503(self, client):
        original = mmm_service.model.is_trained
        mmm_service.model.is_trained = False
        try:
            response = client.post("/predictions", files=spend_files())
            assert response.status_code == 503
        finally:
            mmm_service.model.is_trained = original


class TestModelsEndpoint:
    def test_model_info_includes_training_results(self, client, loaded):
        data = client.get("/models").json()
        assert data["is_trained"] is True
        assert data["model_type"] == config.default_model_type
        if mmm_service.training_summary:
            assert "linear" in data["holdout_metrics"]
            assert set(data["channel_roi"][data["model_type"]]) == set(
                mmm_service.training_summary["spend_cols"])
