import mlflow
import mlflow.sklearn
import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb

from src.data.load_data import load_data
from src.features.build_features import build_features


def train(
    data_path="data/raw/flight_data.csv",
    config_path="configs/train.yaml",
    model_config_path="configs/model.yaml"
):
    # -------------------------------
    # 1️⃣ Load configs
    # -------------------------------
    train_cfg = yaml.safe_load(open(config_path))
    model_cfg = yaml.safe_load(open(model_config_path))

    # -------------------------------
    # 2️⃣ Load and preprocess data
    # -------------------------------
    df = load_data(data_path)
    df = build_features(df)  # returns processed DataFrame

    # Separate features and target
    target_col = "price"
    X = df.drop(columns=[target_col, "flight"], errors="ignore")
    y = df[target_col]

    # -------------------------------
    # 3️⃣ Train/validation split
    # -------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=train_cfg["train"]["test_size"],
        random_state=train_cfg["train"]["random_state"]
    )

    # -------------------------------
    # 4️⃣ Define preprocessing
    # -------------------------------
    categorical = ['airline', 'source_city', 'departure_time', 'arrival_time', 'destination_city']
    ordinal = ['stops', 'class']
    ordinal_categories = [
        ['zero', 'one', 'two_or_more'],   # stops
        ['Economy', 'Business']           # class
    ]
    numerical = [col for col in X.columns if col not in categorical + ordinal]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("ord", OrdinalEncoder(categories=ordinal_categories), ordinal),
        ("num", "passthrough", numerical)
    ])

    # -------------------------------
    # 5️⃣ Define model
    # -------------------------------
    lgb_reg = lgb.LGBMRegressor(**model_cfg["model"]["params"])

    # -------------------------------
    # 6️⃣ Build pipeline with log-target
    # -------------------------------
    pipeline = TransformedTargetRegressor(
        regressor=Pipeline([
            ("preprocessor", preprocessor),
            ("model", lgb_reg)
        ]),
        transformer=FunctionTransformer(np.log1p, inverse_func=np.expm1)
    )

    # -------------------------------
    # 7️⃣ MLflow tracking
    # -------------------------------
    mlflow.set_experiment("flight-price-prediction")

    with mlflow.start_run():
        # Train model
        pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred = pipeline.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)

        print(f"Validation RMSE: {rmse:.4f}")
        print(f"Validation R²: {r2:.4f}")

        # Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # Log model params
        mlflow.log_params(model_cfg["model"]["params"])

        # Log the model to MLflow
        mlflow.sklearn.log_model(pipeline, "flight_price_model")

        # Also save locally for deployment
        joblib.dump(pipeline, "models/pipeline.joblib")
        print("✅ Pipeline saved to models/pipeline.joblib")


if __name__ == "__main__":
    train()
