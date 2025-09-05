from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# ------------------------------
# Load trained pipeline
# ------------------------------
pipeline = joblib.load("models/pipeline.joblib")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------
# Define API
# ------------------------------
app = FastAPI(title="Flight Price Prediction API")

class FlightData(BaseModel):
    airline: str
    source_city: str
    departure_time: str
    arrival_time: str
    destination_city: str
    stops: str
    class_type: str
    duration: float
    days_left: int

@app.get("/")
async def root():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

@app.post("/predict")
def predict(data: FlightData):
    # Convert to DataFrame
    df = pd.DataFrame([data.model_dump()])
    df.rename(columns={"class_type": "class"}, inplace=True)
    # Predict
    price_pred = pipeline.predict(df)
    return {"predicted_price": float(price_pred[0])}

# ------------------------------
# Run with: uvicorn deploy_app:app --reload
# ------------------------------
