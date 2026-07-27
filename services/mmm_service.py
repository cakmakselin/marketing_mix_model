from pathlib import Path
from models.linear_model import LinearMMMModel
from models.bayesian_mmm import BayesianMMMModel
from data.ingestion import DataIngestor
from evaluation.metrics import calculate_mape
from config import config


class MMMService:
    def __init__(self, model_type="linear", adstock_decay=0.0):
        # initialize model based on type
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
        # run data pipeline and train model
        print("Loading data...")
        ingestor = DataIngestor()
        data = ingestor.run()

        # train model
        spend_cols = [col for col in data.columns if col not in ['date', 'sales']]
        self.model.train(data, 'sales', spend_cols)
        self.data = data

    def load_pretrained(self):
        # load pre-trained model from saved files
        models_dir = config.saved_models_path

        if self.model_type == "linear":
            model_file = models_dir / f"trained_{self.model_type}_model.pkl"
            if not model_file.exists():
                print(f"Pre-trained model not found at {model_file}")
                return False
            self.model.load(str(model_file))

        elif self.model_type == "bayesian":
            trace_file = models_dir / f"trained_{self.model_type}_trace.nc"
            if not trace_file.exists():
                print(f"Bayesian trace not found at {trace_file}")
                return False

            try:
                loaded = self.model.load_trace(str(trace_file))
            except Exception as exc:
                print(f"Failed to load pretrained Bayesian model: {exc}")
                return False

            if not loaded:
                return False

        print(f"Pre-trained {self.model_type} model loaded successfully")
        return True

    def predict(self, data=None):
        # make predictions
        if data is None:
            if self.data is None:
                raise ValueError("No data available for prediction")
            data = self.data
        return self.model.predict(data)

    def evaluate(self):
        # evaluate model performance
        if self.data is None:
            raise ValueError("No data available for evaluation")

        predictions = self.predict()
        actual = self.data['sales']
        mape = calculate_mape(actual, predictions)
        return {'mape': mape}
