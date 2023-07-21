from abc import ABC

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class AbstractTransformer(ABC):
    """
    This (abstract) transformer is to be inherited by the other transformers.
    It may not be instantiated on its own. Implements 'fit' method and checks whether 'X' is a DataFrame
    """

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("The input 'X' must be a pandas.DataFrame")
        return self


class TargetTransformer(BaseEstimator, TransformerMixin, AbstractTransformer):
    """Transformer for target variable transformation.

    This transformer calculates the trip duration in minutes from the 'dropoff_datetime' and 'pickup_datetime'
    columns and applies filtering to remove outliers. It is designed to be used specifically for transforming
    the target variable.

    Methods:
    ----------
    transform(X, y=None):
        Transform the target variable by calculating trip duration in minutes and applying outlier filtering.
    """

    def transform(self, X, y=None):
        X_copy = X.copy()
        dropoff_datetime_col = X_copy.columns[
            X_copy.columns.str.contains("dropoff_datetime")
        ].tolist()[0]
        pickup_datetime_col = X_copy.columns[
            X_copy.columns.str.contains("pickup_datetime")
        ].tolist()[0]
        X_copy["duration"] = (
            X_copy[dropoff_datetime_col] - X_copy[pickup_datetime_col]
        ).dt.total_seconds() / 60
        X_copy = X_copy[(X_copy["duration"] >= 1) & (X_copy["duration"] <= 60)]

        return X_copy


class PassengerOutlierTransformer(BaseEstimator, TransformerMixin, AbstractTransformer):
    """A transformer that handles outliers in the 'passenger_count' column.

    Parameters:
    -----------
    None

    Returns:
    --------
    pandas.DataFrame:
        The transformed DataFrame with outliers removed in the 'passenger_count' column.
    """

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_copy = X_copy[
            (X_copy["passenger_count"] > 0) & (X_copy["passenger_count"] <= 8)
        ]

        return X_copy


class FeaturesAndTargetTransformer(
    BaseEstimator, TransformerMixin, AbstractTransformer
):
    def transform(self, X, y=None):
        features = [
            "PULocationID",
            "DOLocationID",
            "trip_distance",
            "passenger_count",
            "fare_amount",
            "total_amount",
        ]
        target = "duration"

        X_copy = X.copy()
        X_copy = X_copy[features + [target]]

        return X_copy
