import json

from models.linear_model import LinearMMMModel
from models.bayesian_mmm import BayesianMMMModel
from data.ingestion import DataIngestor
from evaluation.metrics import evaluate_model
from config import config


class MMMService:
    def __init__(self, model_type="linear", adstock_decay=0.0):
        self.model_type = model_type
        self.adstock_decay = adstock_decay
        #fit at training time, applied to prediction uploads so cleaning
        #behaves the same in both contexts
        self.cleaning_thresholds = None
        self.training_summary = None

        if model_type == "linear":
            self.model = LinearMMMModel(adstock_decay=adstock_decay)
        elif model_type == "bayesian":
            self.model = BayesianMMMModel(adstock_decay=adstock_decay)
        else:
            raise ValueError("model_type must be 'linear' or 'bayesian'")

    def train(self):
        print("Loading data...")
        ingestor = DataIngestor()
        data = ingestor.run(training=True)
        self.cleaning_thresholds = ingestor.cleaning_thresholds

        spend_cols = [col for col in data.columns if col.endswith('_spend')]
        data = data[['date', 'sales'] + spend_cols]
        self.model.train(data, 'sales', spend_cols)

    def load_pretrained(self):
        models_dir = config.saved_models_path

        if self.model_type == "linear":
            model_file = models_dir / "trained_linear_model.pkl"
        else:
            model_file = models_dir / "trained_bayesian_trace.nc"

        try:
            self.model.load(str(model_file))
        except (FileNotFoundError, ImportError, ValueError, KeyError) as exc:
            print(f"Failed to load pretrained {self.model_type} model: {exc}")
            return False

        summary_file = models_dir / "training_summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                self.training_summary = json.load(f)
            self.cleaning_thresholds = self.training_summary.get("cleaning_thresholds")
        else:
            print("Warning: training_summary.json missing; prediction uploads "
                  "will be cleaned with thresholds fit on the upload itself")

        print(f"Pre-trained {self.model_type} model loaded successfully")
        return True

    def ingest_prediction_data(self, data_dir):
        #prediction mode: sales optional, any length, training-time thresholds
        ingestor = DataIngestor(data_dir=data_dir)
        return ingestor.run(training=False, cleaning_thresholds=self.cleaning_thresholds)

    def predict(self, data):
        return self.model.predict(data)

    def predict_interval(self, data, hdi_prob=0.9):
        #credible intervals are only available for the bayesian model with a trace
        if hasattr(self.model, 'predict_interval'):
            return self.model.predict_interval(data, hdi_prob=hdi_prob)
        return None

    def evaluate(self, data):
        #stateless: evaluates predictions against the sales column in `data`
        if 'sales' not in data.columns:
            raise ValueError("Evaluation requires a sales column")
        predictions = self.predict(data)
        return evaluate_model(data['sales'], predictions)
