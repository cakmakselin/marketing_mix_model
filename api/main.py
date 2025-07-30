from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Union
from datetime import date
import sys
import tempfile
import os
import io

#add project root for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import config
from services.mmm_service import MMMService
from data.ingestion import DataIngestor

app = FastAPI(title="Marketing Mix Model API", version="1.0.0")

#initialize MMM service with pre-trained model
print(f"Loading pre-trained {config.default_model_type} model with adstock_decay={config.adstock_decay:.3f}")
mmm_service = MMMService(model_type=config.default_model_type, adstock_decay=config.adstock_decay)
mmm_service.load_pretrained()

class PredictionResponse(BaseModel):
    #sales prediction response
    forecast: List[Dict[str, Union[str, float]]]  # Structured date-prediction pairs
    model_type: str
    adstock_decay: float
    rows_processed: int
    evaluation: Optional[Dict] = None  # Only present if sales data provided

@app.get("/health")
def health_check():
    #health check with service status
    return {
        "status": "healthy",
        "service_ready": mmm_service.model.is_trained,
        "model_type": config.default_model_type,
        "adstock_decay": mmm_service.model.adstock_decay
    }

@app.post("/predictions", response_model=PredictionResponse)
async def create_prediction_from_csvs(files: List[UploadFile] = File(...)):
    #create predictions from multiple CSV files (one per channel)
    if not mmm_service.model.is_trained:
        raise HTTPException(status_code=400, detail="Model not trained")
    
    try:
        #create temporary directory and save files for DataIngestor
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            #save uploaded files to temp directory
            for file in files:
                contents = await file.read()
                file_path = temp_path / file.filename
                
                #write file contents
                with open(file_path, 'wb') as f:
                    f.write(contents)
            
            ingestor = DataIngestor(data_dir=temp_path)
            merged_data = ingestor.run()
            
            #check if sales data is present for evaluation
            has_sales = 'sales' in merged_data.columns
            evaluation_result = None
            if has_sales:
                #temporarily set service data for evaluation
                original_data = mmm_service.data
                mmm_service.data = merged_data
                
                try:
                    #use existing service evaluate method
                    evaluation_result = mmm_service.evaluate()
                except Exception as e:
                    print(f"Evaluation failed: {e}")
                    evaluation_result = {"error": "Evaluation failed"}
                finally:
                    #restore original data
                    mmm_service.data = original_data
            
            #make predictions
            predictions = mmm_service.model.predict(merged_data)
            
            #handle both numpy arrays and lists
            if hasattr(predictions, 'tolist'):
                predictions_list = predictions.tolist()
            else:
                predictions_list = predictions
            
            return PredictionResponse(
                forecast=[{"date": str(d), "predicted_sales": p} for d, p in zip(merged_data['date'].dt.date.tolist(), predictions_list)],
                model_type=config.default_model_type,
                adstock_decay=mmm_service.model.adstock_decay,
                rows_processed=len(merged_data),
                evaluation=evaluation_result
            )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

@app.get("/models")
def get_model_info():
    #get current model information
    return {
        "model_type": config.default_model_type,
        "adstock_decay": mmm_service.model.adstock_decay,
        "is_trained": mmm_service.model.is_trained
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 