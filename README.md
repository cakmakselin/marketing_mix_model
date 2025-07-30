# Marketing Mix Model (MMM)

The system can:
- Analyze marketing performance across 6 channels (TV, radio, social media, search, outdoor, print)
- Use either linear regression or Bayesian modeling 
- Serve predictions via a REST API with CSV file uploads

## Quick Start

### Prerequisites
- Python 3.9+ (I used 3.9.13)
- A virtual environment

### Setup Instructions

1. **Clone and enter the project**
```bash
git clone https://github.com/cakmakselin/marketing_mix_model)
cd marketing_mix_model
```

2. **Create and activate virtual environment**
```bash
python -m venv venv
source venv/bin/activate 
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Train models**
```bash
python scripts/train_models.py
```

5. **Start the API server**
```bash
python api/main.py
```
The API will be available at `http://localhost:8000` (SwaggerUI at `/docs`)

### Running Tests
```bash
python -m pytest tests/ -v
```

## System Architecture

Here's a general overview of the codebase. I tried to keep it modular so it's easier to understand and extend:

```
marketing_mix_model/
├── data/                    # Data pipeline components
│   ├── ingestion.py        # Loads and combines raw data files
│   ├── validation.py       # Checks data quality and formats
│   └── storage.py          # Handles processed data persistence
├── models/                  # The ML models
│   ├── base_model.py       # Base class with adstock transformations
│   ├── linear_model.py     # Scikit-learn linear regression approach
│   ├── bayesian_mmm.py     # PyMC Bayesian modeling approach
│   └── saved_models/       # Trained model artifacts
├── evaluation/              # Model evaluation
│   └── metrics.py          # MAPE and other evaluation metrics
├── services/                # Business logic layer
│   └── mmm_service.py      # Main service that orchestrates everything
├── scripts/                 # Utility scripts
│   └── train_models.py     # Explicit model training script
├── api/                     # REST API layer
│   └── main.py             # FastAPI application
└── tests/                   # Test suite
```

### Core Components

**Data Pipeline** (`data/`)
- **DataIngestor**: Finds and loads marketing spend + sales data automatically
- **DataValidator**: Makes sure data quality is good (no missing dates, proper formats, etc.)
- **Storage**: Can save/load processed data to parquet files (mainly used during training)

**Models** (`models/`)
- **BaseMMMModel**: Shared functionality like adstock transformation and feature engineering
- **LinearMMMModel**: Fast sklearn-based linear regression with pickle serialization
- **BayesianMMMModel**: More sophisticated PyMC model with Arviz trace storage

**Evaluation** (`evaluation/`)
- Uses MAPE (Mean Absolute Percentage Error) to evaluate model performance on validation data

**Service Layer** (`services/`)
- **MMMService**: Main business logic that handles training, model loading, and prediction workflows
- Keeps the API simple by hiding model complexity

**API** (`api/`)
- FastAPI REST interface for CSV file uploads and predictions
- Loads pre-trained models on startup using the configured model type
- Returns structured forecasts with date-prediction pairs

## API Usage

### Upload CSV Files for Predictions

The API accepts multiple CSV files (one per marketing channel) via the `/predictions` endpoint:

**Required CSV Format:**
- Each file needs a `date` column and either `spend` or `sales` column
- Date format: YYYY-MM-DD
- File naming: `{channel}_spend.csv` or `sales_data.csv`

**Example Request:**
Upload files like:
- `tv_spend.csv`
- `radio_spend.csv` 
- `social_media_spend.csv`
- `sales_data.csv` (optional - for evaluation)

**Response Format:**
```json
{
  "forecast": [
    {"date": "2024-01-01", "predicted_sales": 195000.5},
    {"date": "2024-01-02", "predicted_sales": 198500.2}
  ],
  "model_type": "bayesian",
  "adstock_decay": 0.3,
  "rows_processed": 365,
  "evaluation": {"mape": 10.8}  
}
```

### Available Endpoints

- `GET /health` - Service health check
- `POST /predictions` - Upload CSV files for batch predictions
- `GET /models` - Get current model information

## Technical Approach & Design Decisions


Since I didn't have much background info about the specific business or marketing context, I focused on understanding the general structure of MMM problems and familiarizing myself with the topic. The main challenge with marketing data is that it doesn't follow typical ML assumptions. It's seasonal, has campaign-driven spikes, and different channels operate on completely different scales. Rather than force standard techniques, I analyzed the specific patterns in the data first, then chose transformations and architectures that directly addressed those characteristics.


### Exploratory Data Analysis (`EDA`)

**Descriptive Statistics**
Started with basic shape and coverage checks (365 rows × 7 columns covering a full year). The descriptive statistics immediately revealed serious data quality issues: negative values across multiple channels and extreme outliers with max values wildly disproportionate to means.

**Distribution: Box Plots**  
The medians revealed distinct spending patterns. Radio, outdoor, TV, and print all had median values of 0, meaning over half the days had zero spend for these channels. Social media and search had non-zero medians, indicating more consistent daily spending. 

**Time Series Analysis**
The raw time series plots were dominated by extreme outliers that made analysis difficult. Social media, search, outdoor, and sales were nearly impossible to interpret. the massive spikes (283k, 225k, 1.2M, 7M respectively) compressed normal spending into flat lines at the bottom. Only radio, TV, and print showed clear campaign patterns because their outliers were less extreme relative to normal ranges.

This visualization problem demonstrates the need for outlier removal. Under the assumption that deviations of this magnitude are data quality issues, negative spending values likely represent data entry errors in a marketing context, and extreme highs (75x to 132x normal levels) are more consistent with mistakes than actual campaign spending.

**Data Preprocessing Steps**
- Flagged extreme highs >20x the 95th percentile as likely data errors while keeping normal campaign spikes
- Removed negative values since you can't spend negative money or have negative sales  
- Flagged zero sales days as unrealistic
- Handled existing missing values throughout the dataset
All flagged values were replaced with NaN and linearly interpolated to keep the time series intact for modeling. Linear interpolation was chosen because it's simple and doesn't introduce artificial patterns that could mess with the analysis.

**Validation (Post-Cleaning Correlations)**
After implementing preprocessing rules, meaningful patterns emerged. TV spend showed strongest sales correlation (0.50), followed by outdoor (0.37) and radio (0.21).

### Data Pipeline (`data/`)

**File structure handling**: Rather than hardcode filenames, I used glob patterns to automatically discover spend files. This came from realizing that marketing teams often add new channels or change naming conventions. The date-based outer join handles the reality that different channels might have different reporting periods.

**Validation strategy**: Added Pydantic validators at each pipeline stage because I kept running into silent failures. Data would look fine until the model training crashed with cryptic errors. Early validation with structured error messages made debugging much faster.

### Models Architecture (`models/`)

**Feature engineering decisions**: Applied adstock transformation to spend columns to model decaying marketing effects. This made sense for the data as well since spend channels showed patterns like 50% zero-spend days, then bursts. Sales data doesn't have carryover effects, so no adstock there.

Log transforms addressed multiple issues I found: 
- extreme scale differences 
- diminishing returns modeling (higher spending should have proportionally less impact) 
- better performance for Bayesian sampling which works better with smaller-scale values

**Model choice**: I implemented both linear regression and Bayesian approaches to cover different use cases. Linear regression (scikit-learn) is fast, interpretable, and a good baseline. The Bayesian model (PyMC) handles uncertainty better. The config file specifies which model type to use.

**Simplifications**: I went with a single adstock decay for all channels and didn't use fancy priors, which is definitely simpler than what you'd see in production MMM systems. In reality, TV probably has way different carryover effects than Google Ads, and if you had a marketing expert they could tell you what to expect from each channel. But since I don't have that domain knowledge, I figured it's better to keep things simple than pretend I know stuff I don't. This approach works well for getting started or when you're exploring data without strong assumptions about how different channels should behave.

**Inheritance structure**: Both model types needed identical feature engineering but different training logic. A base class avoided code duplication while keeping model-specific implementations separate. This made it easier to experiment with different models without rewriting the preprocessing.

### API Design (`api/`)

**CSV-only approach**: Decided to focus on CSV file uploads since that's how marketing data typically exists. This is more practical than complex JSON structures for real-world usage.

**Service layer separation**: Put business logic in MMMService rather than directly in API endpoints. This came from wanting to keep the API focused on request/response handling while making the core functionality reusable for batch processing or different interfaces.

**Pre-trained model loading**: API loads saved models on startup instead of training, making it production-ready. The service handles all model loading complexity internally.



## Future Improvements

**Models & Analysis**
- seasonality and external factors (holidays, weather, economic indicators)
- scenario planning features
- add ROI simulations for budget optimization
- Include business-specific priors when domain experts are available

**Technical Infrastructure**
- Containerize with Docker for easier deployment
- Connect to BigQuery for real-time ingestion
- Deploy on Google Vertex AI for scalable training
- Add model versioning and automated retraining pipelines

## Tech Stack

- **Core ML**: scikit-learn, PyMC, NumPy, pandas
- **API**: FastAPI + uvicorn
- **Testing**: pytest
- **Data**: Handles CSV inputs, supports parquet for processed data
