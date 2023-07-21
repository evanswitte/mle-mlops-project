import argparse
import base64
import os

import mlflow
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from src.preprocessor_nyc_taxi import PreprocessorNycTaxiData

parser = argparse.ArgumentParser()
parser.add_argument(
    "--cml_run", default=False, action=argparse.BooleanOptionalAction, required=True
)
args = parser.parse_args()
cml_run = args.cml_run

GOOGLE_APPLICATION_CREDENTIALS = "./credentials.json"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS


# Set variables
color = "green"
year = 2022
month = 1
features = [
    "PULocationID",
    "DOLocationID",
    "trip_distance",
    "passenger_count",
    "fare_amount",
    "total_amount",
]
target = "duration"
model_name = f"{color}-taxi-trip-duration-lr"

# Set up the connection to MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Setup the MLflow experiment
mlflow.set_experiment(model_name)


PREPROCESSOR = PreprocessorNycTaxiData()

df = PREPROCESSOR.load_data_to_dataframe(color, month, year)

df_processed = PREPROCESSOR.preprocess_fit_transform(df)

y = df_processed["duration"]
X = df_processed.drop(columns=["duration"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2
)


with mlflow.start_run() as run:
    tags = {
        "model": "linear regression",
        "developer": "Evo",
        "dataset": f"{color}-taxi",
        "year": year,
        "month": month,
        "features": features,
        "target": target,
    }
    mlflow.set_tags(tags)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred_train = lr.predict(X_train)
    rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
    mlflow.log_metric("rmse", rmse_train)

    y_pred_test = lr.predict(X_test)
    rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
    mlflow.log_metric("rmse test", rmse_test)

    mlflow.sklearn.log_model(lr, "model")

    run_id = run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri=model_uri, name=model_name)


if cml_run:
    with open("metrics.txt", "w") as f:
        f.write(f"RMSE on the Train Set: {rmse_train}")
        f.write(f"RMSE on the Test Set: {rmse_test}")
