from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from pathlib import Path
from typing import List, Optional, Dict, Union
import sys
import tempfile

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import config
from services.mmm_service import MMMService

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading pre-trained {config.default_model_type} model with adstock_decay={config.adstock_decay:.3f}")
    mmm_service.load_pretrained()
    yield


app = FastAPI(title="Marketing Mix Model API", version="1.0.0", lifespan=lifespan)

mmm_service = MMMService(model_type=config.default_model_type, adstock_decay=config.adstock_decay)

@app.get("/")
def root():
    # root endpoint with API information
    return {
        "title": "Marketing Mix Model API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health (GET) - health check and service status",
            "predictions": "/predictions (POST) - get predictions from CSV files",
            "models": "/models (GET) - model info, holdout metrics, channel ROI",
            "docs": "/docs - interactive API documentation"
        }
    }

class PredictionResponse(BaseModel):
    #sales prediction response; interval bounds only present for bayesian model
    forecast: List[Dict[str, Union[str, float]]]
    model_type: str
    adstock_decay: float
    rows_processed: int
    evaluation: Optional[Dict] = None  # Only present if sales data provided

@app.get("/health")
def health_check():
    #degraded (not "healthy") when no model could be loaded
    ready = mmm_service.model.is_trained
    return {
        "status": "ok" if ready else "degraded",
        "service_ready": ready,
        "model_type": config.default_model_type,
        "adstock_decay": mmm_service.model.adstock_decay
    }

@app.post("/predictions", response_model=PredictionResponse)
async def create_prediction_from_csvs(files: List[UploadFile] = File(...)):
    #create predictions from multiple CSV files (one per channel);
    #sales_data.csv is optional and only used for evaluation
    if not mmm_service.model.is_trained:
        raise HTTPException(status_code=503, detail="Model not trained")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            for file in files:
                contents = await file.read()
                with open(temp_path / file.filename, 'wb') as f:
                    f.write(contents)

            merged_data = mmm_service.ingest_prediction_data(temp_path)

            #evaluate against actuals when provided (stateless, no service mutation)
            evaluation_result = None
            if 'sales' in merged_data.columns:
                try:
                    evaluation_result = mmm_service.evaluate(merged_data)
                except Exception as e:
                    print(f"Evaluation failed: {e}")
                    evaluation_result = {"error": "Evaluation failed"}

            predictions = mmm_service.predict(merged_data)
            interval = mmm_service.predict_interval(merged_data)

            forecast = []
            for i, (d, p) in enumerate(zip(merged_data['date'].dt.date, predictions)):
                row = {"date": str(d), "predicted_sales": float(p)}
                if interval is not None:
                    row["lower_90"] = float(interval[0][i])
                    row["upper_90"] = float(interval[1][i])
                forecast.append(row)

            return PredictionResponse(
                forecast=forecast,
                model_type=config.default_model_type,
                adstock_decay=mmm_service.model.adstock_decay,
                rows_processed=len(merged_data),
                evaluation=evaluation_result
            )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

@app.get("/models")
def get_model_info():
    #model info plus training-time results (holdout metrics, contributions, ROI)
    info = {
        "model_type": config.default_model_type,
        "adstock_decay": mmm_service.model.adstock_decay,
        "is_trained": mmm_service.model.is_trained,
    }
    if mmm_service.training_summary:
        summary = mmm_service.training_summary
        info["holdout_metrics"] = summary.get("holdout")
        info["channel_contributions"] = summary.get("channel_contributions")
        info["channel_roi"] = summary.get("channel_roi")
        info["train_period"] = summary.get("train_period")
    return info

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
