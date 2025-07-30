from pathlib import Path
from models.linear_model import LinearMMMModel
from models.bayesian_mmm import BayesianMMMModel
from data.ingestion import DataIngestor
from evaluation.metrics import calculate_mape

class MMMService:
    def __init__(self, model_type="linear", adstock_decay=0.0):
        #initialize model based on type
        self.model_type = model_type
        self.adstock_decay = adstock_decay
        self.data = None 
        
        if model_type == "linear":
            self.model = LinearMMMModel(adstock_decay=adstock_decay)
        elif model_type == "bayesian":
            self.model = BayesianMMMModel(adstock_decay=adstock_decay)
        else:
            raise ValueError("model_type must be 'linear' or 'bayesian'")
    
    def train(self):
        #run data pipeline and train model
        print("Loading data...")
        ingestor = DataIngestor()
        data = ingestor.run()
        
        #train model
        spend_cols = [col for col in data.columns if col not in ['date', 'sales']]
        self.model.train(data, 'sales', spend_cols)
        self.data = data
    
    def load_pretrained(self):
        #load pre-trained model from saved files
        models_dir = Path("models/saved_models")
        
        if self.model_type == "linear":
            model_file = models_dir / f"trained_{self.model_type}_model.pkl"
            if not model_file.exists():
                raise FileNotFoundError(f"Pre-trained model not found: {model_file}. Run 'python scripts/train_models.py' first.")
            self.model.load(str(model_file))
            
        elif self.model_type == "bayesian":
            trace_file = models_dir / f"trained_{self.model_type}_trace.nc"
            if not trace_file.exists():
                raise FileNotFoundError(f"Bayesian trace not found: {trace_file}. Run 'python scripts/train_models.py' first.")
            self.model.load_trace(str(trace_file))
            
        print(f"Pre-trained {self.model_type} model loaded successfully")
    
    def predict(self, data=None):
        #make predictions
        if data is None:
            data = self.data
        return self.model.predict(data)
    
    def evaluate(self):
        #evaluate model performance
        predictions = self.predict()
        actual = self.data['sales']
        mape = calculate_mape(actual, predictions)
        return {'mape': mape}
        

        
