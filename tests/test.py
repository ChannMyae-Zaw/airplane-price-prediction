import joblib
import pandas as pd

pipeline = joblib.load("models/pipeline.joblib")

# Load some test data
df_test = pd.read_csv("data/raw/flight_data.csv")  # or a slice of your dataset
X_test = df_test.drop(columns=["price", "flight"], errors="ignore")

# Make predictions
y_pred = pipeline.predict(X_test)
y_true = df_test["price"].iloc[915:925]  # select rows 915–924
y_pred_slice = y_pred[915:925]

print(y_pred_slice - y_true)
