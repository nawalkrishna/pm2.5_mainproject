"""Baseline models with one interface: ``fit(train_df, y)`` / ``predict(df)`` (Stage 5, task 2).

All preprocessing is fitted on the training rows only (imputer medians, scaler, climatology).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class Persistence:
    """Yesterday's PM2.5 at the same station (needs pm25_daily_lag1)."""

    def fit(self, df: pd.DataFrame, y: pd.Series) -> "Persistence":
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["pm25_daily_lag1"].to_numpy(float)


class Climatology:
    """Mean of the training target by station x month; unseen station -> state x month -> month."""

    def fit(self, df: pd.DataFrame, y: pd.Series) -> "Climatology":
        t = df[["station_id", "state", "month"]].assign(y=y.to_numpy())
        self.st = t.groupby(["station_id", "month"])["y"].mean()
        self.sm = t.groupby(["state", "month"])["y"].mean()
        self.m = t.groupby("month")["y"].mean()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        p = pd.Series(self.st.reindex(pd.MultiIndex.from_frame(df[["station_id", "month"]])).to_numpy(), index=df.index)
        p = p.fillna(pd.Series(self.sm.reindex(pd.MultiIndex.from_frame(df[["state", "month"]])).to_numpy(), index=df.index))
        return p.fillna(df["month"].map(self.m)).to_numpy(float)


class Linear:
    """OLS on median-imputed, standardised features (imputer and scaler fitted on train)."""

    def __init__(self, features: list[str], impute: str = "median"):
        self.features = features
        self.model = make_pipeline(SimpleImputer(strategy=impute, keep_empty_features=True), StandardScaler(),
                                   LinearRegression())

    def fit(self, df, y):
        self.model.fit(df[self.features], y)
        return self

    def predict(self, df):
        return self.model.predict(df[self.features])


class RandomForest:
    """Random Forest using native NaN support (no imputation, task 7)."""

    def __init__(self, features: list[str], seed: int, **params):
        self.features = features
        self.model = RandomForestRegressor(random_state=seed, n_jobs=-1, **params)

    def fit(self, df, y):
        self.model.fit(df[self.features], y)
        return self

    def predict(self, df):
        return self.model.predict(df[self.features])
