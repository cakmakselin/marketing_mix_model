import pandas as pd
import numpy as np


class BaseMMMModel:
    def __init__(self, adstock_decay=0.0):
        self.adstock_decay = adstock_decay
        self.is_trained = False
        self.feature_cols = []
        self.spend_cols = []

    def apply_adstock(self, spend_series, decay_rate):
        #geometric adstock: effect today = spend today + decay * effect yesterday
        spend_values = np.asarray(spend_series, dtype=np.float64)
        if decay_rate <= 0:
            return spend_values

        adstocked = np.zeros_like(spend_values)
        adstocked[0] = spend_values[0]
        for i in range(1, len(spend_values)):
            adstocked[i] = spend_values[i] + decay_rate * adstocked[i - 1]
        return adstocked

    def apply_log_transform(self, series_or_array):
        #log transformation for diminishing returns
        return np.log1p(np.asarray(series_or_array, dtype=np.float64))

    def build_features(self, df, spend_cols):
        #one feature per channel: log1p(adstock(spend)) — carryover + diminishing returns.
        #features are only ever derived from spend columns, never from sales,
        #so the target cannot leak into the design matrix.
        missing = [col for col in spend_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing spend columns in input data: {missing}")

        if 'date' not in df.columns:
            raise ValueError("Input data must have a 'date' column")

        features = pd.DataFrame(index=df.index)
        for col in spend_cols:
            adstocked = self.apply_adstock(df[col], self.adstock_decay)
            features[f"{col}_adstock_log"] = self.apply_log_transform(adstocked)

        #seasonal baseline: annual sin/cos wave so channel coefficients don't
        #absorb seasonality (monthly sales swing 134k-226k in this data).
        #a smooth wave extrapolates to unseen months, unlike month dummies;
        #a separate trend term is not identifiable with one year of data.
        day_of_year = pd.to_datetime(df['date']).dt.dayofyear.values
        features['season_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
        features['season_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)

        return features

    def raw_coefficients(self):
        #(intercept, coefs) on the raw feature scale, aligned with feature_cols.
        #implemented per subclass so contributions work for both model types
        raise NotImplementedError

    def channel_contributions(self, df):
        #additive decomposition: predicted sales attributable to each channel
        #over the given period (model is linear in its features)
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")
        X = self.build_features(df, self.spend_cols)[self.feature_cols].values
        _, coefs = self.raw_coefficients()

        contributions = {}
        for i, col in enumerate(self.feature_cols):
            if col.endswith('_adstock_log'):
                channel = col[: -len('_adstock_log')]
                contributions[channel] = float((X[:, i] * coefs[i]).sum())
        return contributions

    def channel_roi(self, df):
        #attributed sales per unit of spend over the given period
        contributions = self.channel_contributions(df)
        roi = {}
        for channel, contribution in contributions.items():
            spent = float(df[channel].sum())
            roi[channel] = contribution / spent if spent > 0 else None
        return roi

    #to be implemented in subclasses
    def train(self, df, sales_col, spend_cols): raise NotImplementedError
    def predict(self, data): raise NotImplementedError
