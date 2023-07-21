import os

import pandas as pd
from sklearn.pipeline import Pipeline

from nyc_feature_engineering import (
    FeaturesAndTargetTransformer,
    PassengerOutlierTransformer,
    TargetTransformer,
)


class PreprocessorNycTaxiData:
    """Preprocessing pipeline for NYC Taxi Data.

    This class encapsulates the preprocessing steps for NYC Taxi Data. It consists of two main pipelines:
    the data cleaning pipeline and the feature engineering pipeline.

    Methods:
    ----------
    load_data_to_dataframe(color, month, year):
        Load NYC Taxi data from a given color, month, and year into a DataFrame.
    preprocess_fit_transform(df):
        Fit and transform the input DataFrame using the preprocessing pipeline.
    preprocess_transform(df):
        Transform the input DataFrame using the preprocessing pipeline.
    """

    def __init__(self):
        self.preprocessor_pipe = Pipeline(
            [
                ("target", TargetTransformer()),
                ("passenger_count", PassengerOutlierTransformer()),
                ("featureTarget", FeaturesAndTargetTransformer()),
            ]
        )

    def load_data_to_dataframe(self, color: str, month: int, year: int) -> pd.DataFrame:
        """Load NYC Taxi data from a given color, month, and year into a DataFrame.

        Parameters:
        ----------
        color : str
            Color of the taxi.

        month : int
            Month of the data.

        year : int
            Year of the data.

        Returns:
        ----------
        df : DataFrame
            Loaded DataFrame from the given color, month, and year."""
        data_dir = f"../data/{year}"

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{color}_tripdata_{year}-{month:02d}.parquet"

        os.system(f"wget -N -P {data_dir} {url}")

        file_name = os.path.join(
            data_dir, f"{color}_tripdata_{year}-{month:02d}.parquet"
        )

        df = pd.read_parquet(file_name)

        return df

    def preprocess_fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform the input DataFrame using the preprocessing pipeline.

        This method fits the preprocessing pipeline to the input DataFrame and applies the transformations.

        Parameters:
        ----------
        df : DataFrame
            Input DataFrame.

        Returns:
        ----------
        preprocessed_df : DataFrame
            Preprocessed DataFrame after fitting and transforming the data.
        """
        return self.preprocessor_pipe.fit_transform(df)

    def preprocess_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the input DataFrame using the preprocessing pipeline.

        This method applies the transformations from the preprocessing pipeline to the input DataFrame.

        Parameters:
        ----------
        df : DataFrame
            Input DataFrame.

        Returns:
        ----------
        preprocessed_df : DataFrame
            Preprocessed DataFrame after transforming the data.
        """
        return self.preprocessor_pipe.transform(df)
