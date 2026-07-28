import pytest
import pandas as pd
import numpy as np
from models.linear_model import LinearMMMModel
from models.base_model import BaseMMMModel
from models.bayesian_mmm import BayesianMMMModel


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
        result = model.apply_adstock(pd.Series([100, 0, 0]), 0.0)
        assert list(result) == [100, 0, 0]

    def test_apply_log_transform(self):
        model = BaseMMMModel()
        result = model.apply_log_transform(pd.Series([0, 100, 1000]))
        assert result[0] == 0
        assert result[1] == pytest.approx(4.6, rel=0.1)
        assert result[2] == pytest.approx(6.9, rel=0.1)

    def test_build_features_columns(self, sample_training_data):
        model = BaseMMMModel(adstock_decay=0.2)
        df, spend_cols = sample_training_data
        features = model.build_features(df, spend_cols)

        expected = [f"{c}_adstock_log" for c in spend_cols] + ['season_sin', 'season_cos']
        assert list(features.columns) == expected

    def test_no_feature_derived_from_sales(self, sample_training_data):
        # regression test for target leakage: features must never touch sales
        model = BaseMMMModel(adstock_decay=0.2)
        df, spend_cols = sample_training_data
        features = model.build_features(df, spend_cols)
        assert not any('sales' in col for col in features.columns)

        # perturbing sales must not change any feature
        df2 = df.copy()
        df2['sales'] = df2['sales'] * 100
        features2 = model.build_features(df2, spend_cols)
        pd.testing.assert_frame_equal(features, features2)

    def test_build_features_missing_channel_raises(self, sample_training_data):
        model = BaseMMMModel(adstock_decay=0.2)
        df, spend_cols = sample_training_data
        with pytest.raises(ValueError, match="Missing spend columns"):
            model.build_features(df.drop(columns=[spend_cols[0]]), spend_cols)


class TestLinearMMMModel:
    def test_train_and_predict(self, sample_training_data):
        model = LinearMMMModel(adstock_decay=0.3)
        df, spend_cols = sample_training_data
        model.train(df, 'sales', spend_cols)

        assert model.is_trained
        assert len(model.feature_cols) == len(spend_cols) + 2  # + season sin/cos

        predictions = model.predict(df)
        assert len(predictions) == len(df)

    def test_predict_without_sales_column(self, sample_training_data):
        # the API use case: uploads have spend but no sales
        model = LinearMMMModel(adstock_decay=0.3)
        df, spend_cols = sample_training_data
        model.train(df, 'sales', spend_cols)

        predictions = model.predict(df.drop(columns=['sales']))
        assert len(predictions) == len(df)

    def test_predict_without_training(self, sample_training_data):
        model = LinearMMMModel()
        df, _ = sample_training_data
        with pytest.raises(RuntimeError, match="must be trained"):
            model.predict(df)

    def test_model_learns_signal(self):
        # sales generated from a known channel: the model should recover fit
        np.random.seed(0)
        n = 200
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=n),
            'tv_spend': np.random.uniform(0, 1000, n),
            'radio_spend': np.random.uniform(0, 500, n),
        })
        df['sales'] = 5000 + 800 * np.log1p(df['tv_spend']) + np.random.normal(0, 100, n)

        model = LinearMMMModel(adstock_decay=0.0)
        model.train(df, 'sales', ['tv_spend', 'radio_spend'])
        contributions = model.channel_contributions(df)
        assert contributions['tv_spend'] > 10 * abs(contributions['radio_spend'])

    def test_save_and_load(self, tmp_path, sample_training_data):
        model = LinearMMMModel(adstock_decay=0.3)
        df, spend_cols = sample_training_data
        model.train(df, 'sales', spend_cols)

        filepath = tmp_path / 'linear_model.pkl'
        model.save(str(filepath))

        loaded = LinearMMMModel(adstock_decay=0.1)
        loaded.load(str(filepath))

        assert loaded.is_trained
        assert loaded.adstock_decay == 0.3
        assert loaded.feature_cols == model.feature_cols
        np.testing.assert_allclose(loaded.predict(df), model.predict(df))

    def test_load_rejects_legacy_format(self, tmp_path):
        import pickle
        filepath = tmp_path / 'legacy.pkl'
        with open(filepath, 'wb') as f:
            pickle.dump("not a payload dict", f)
        with pytest.raises(ValueError, match="Unsupported model file format"):
            LinearMMMModel().load(str(filepath))

    def test_channel_roi(self, sample_training_data):
        model = LinearMMMModel(adstock_decay=0.3)
        df, spend_cols = sample_training_data
        model.train(df, 'sales', spend_cols)
        roi = model.channel_roi(df)
        assert set(roi.keys()) == set(spend_cols)


class TestBayesianMMMModel:
    def test_initialization(self):
        model = BayesianMMMModel(adstock_decay=0.5)
        assert model.adstock_decay == 0.5
        assert not model.is_trained
        assert model.trace is None

    def test_predict_without_training(self, sample_training_data):
        model = BayesianMMMModel()
        df, _ = sample_training_data
        with pytest.raises(RuntimeError, match="must be trained"):
            model.predict(df)

    def test_train_predict_save_load(self, tmp_path, sample_training_data):
        # real sampling, no mocks: small model is fast enough
        pytest.importorskip("pymc")
        df, spend_cols = sample_training_data

        model = BayesianMMMModel(adstock_decay=0.3)
        model.train(df, 'sales', spend_cols, draws=100)
        assert model.is_trained

        predictions = model.predict(df.drop(columns=['sales']))
        assert len(predictions) == len(df)

        interval = model.predict_interval(df, hdi_prob=0.9)
        assert interval.shape == (2, len(df))
        assert (interval[0] <= interval[1]).all()

        trace_path = tmp_path / 'trace.nc'
        model.save(str(trace_path))
        assert trace_path.exists()
        assert trace_path.with_suffix('.json').exists()

        loaded = BayesianMMMModel()
        loaded.load(str(trace_path))
        assert loaded.is_trained
        assert loaded.adstock_decay == 0.3
        np.testing.assert_allclose(loaded.predict(df), model.predict(df), rtol=1e-6)

    def test_load_without_metadata_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="metadata not found"):
            BayesianMMMModel().load(str(tmp_path / 'missing.nc'))


# Fixtures
@pytest.fixture
def sample_training_data():
    """Create sample data for model training"""
    np.random.seed(42)

    df = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=60),
        'sales': np.random.normal(1000, 100, 60),
        'tv_spend': np.random.normal(500, 50, 60),
        'radio_spend': np.random.normal(200, 20, 60),
        'social_media_spend': np.random.normal(100, 10, 60)
    })

    for col in ['sales', 'tv_spend', 'radio_spend', 'social_media_spend']:
        df[col] = df[col].abs()

    spend_cols = ['tv_spend', 'radio_spend', 'social_media_spend']

    return df, spend_cols
