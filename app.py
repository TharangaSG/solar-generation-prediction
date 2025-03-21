from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import pandas as pd
from typing import List, Optional
import uvicorn
import os
from src.config import settings

app = FastAPI(
    title="Solar Power Prediction API",
    description="API for predicting solar panel power output based on environmental data",
    version="1.0.0"
)

# Define file paths

# SCALER_PATH = "models\scaler.pkl" 
# MODEL_PATH = "models\spfnet_final_model.keras"

SCALER_PATH = os.path.join("models", "scaler.pkl")
MODEL_PATH = os.path.join("models", "spfnet_final_model.keras")
# Load the model and scaler at startup
try:
    scaler = joblib.load(SCALER_PATH)
    model = load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model or scaler: {str(e)}")

# Define feature names
FEATURES = [
    'Board temperature_℃', 'Radiation intensity_w', 'Wind pressure_kgm2',
    'Top wind speed_MS-2', 'Low wind speed_MS-2', 'station pressure',
    'sea level pressure', 'temperature', 'humidity', 'precipitation',
    'cloud amount', 'irradiance', 'pca_1', 'pca_2', 'pca_3', 'pca_4'
]

# Input data models
class SinglePredictionInput(BaseModel):
    board_temperature: float
    radiation_intensity: float
    wind_pressure: float
    top_wind_speed: float
    low_wind_speed: float
    station_pressure: float
    sea_level_pressure: float
    temperature: float
    humidity: float
    precipitation: float
    cloud_amount: float
    irradiance: float
    pca_1: float
    pca_2: float
    pca_3: float
    pca_4: float

class BatchPredictionInput(BaseModel):
    data: List[SinglePredictionInput]

# Output data models
class PredictionOutput(BaseModel):
    predicted_power: float

class BatchPredictionOutput(BaseModel):
    predictions: List[float]

class CSVPredictionOutput(BaseModel):
    file_path: str
    success: bool
    message: str

# Helper functions
def prepare_input_data(input_data: SinglePredictionInput) -> np.ndarray:
    """Convert a Pydantic model to a NumPy array in the correct order for prediction."""
    return np.array([[
        input_data.board_temperature,
        input_data.radiation_intensity,
        input_data.wind_pressure,
        input_data.top_wind_speed,
        input_data.low_wind_speed,
        input_data.station_pressure,
        input_data.sea_level_pressure,
        input_data.temperature,
        input_data.humidity,
        input_data.precipitation,
        input_data.cloud_amount,
        input_data.irradiance,
        input_data.pca_1,
        input_data.pca_2,
        input_data.pca_3,
        input_data.pca_4
    ]])

# Endpoints
@app.get("/")
async def root():
    return {"message": "Solar Power Prediction API is running. Use /docs for API documentation."}

@app.post("/predict/single", response_model=PredictionOutput)
async def predict_single(input_data: SinglePredictionInput):
    """Predict power output for a single set of features."""
    try:
        # Prepare input
        new_input = prepare_input_data(input_data)
        
        # Scale input
        new_input_scaled = scaler.transform(new_input)
        
        # Make prediction
        prediction = model.predict(new_input_scaled)
        
        return {"predicted_power": float(prediction[0][0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionOutput)
async def predict_batch(input_data: BatchPredictionInput):
    """Predict power output for multiple sets of features."""
    try:
        # Prepare inputs
        inputs = []
        for item in input_data.data:
            inputs.append(prepare_input_data(item)[0])  # Extract the row from the 2D array
        
        batch_input = np.array(inputs)
        
        # Scale inputs
        batch_input_scaled = scaler.transform(batch_input)
        
        # Make predictions
        predictions = model.predict(batch_input_scaled)
        
        # Convert to list of floats
        prediction_list = [float(pred[0]) for pred in predictions]
        
        return {"predictions": prediction_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.post("/predict/csv", response_model=CSVPredictionOutput)
async def predict_from_csv(input_path: str, output_path: Optional[str] = None):
    """Process a CSV file and generate predictions."""
    if not output_path:
        output_path = "predictions.csv"
    
    try:
        # Read input CSV
        df = pd.read_csv(input_path)
        
        # Check if month column exists and drop it
        if "month" in df.columns:
            df = df.drop(["month"], axis=1)
        
        # Ensure all required features are present
        missing_features = [feature for feature in FEATURES if feature not in df.columns]
        if missing_features:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required features in CSV: {', '.join(missing_features)}"
            )
        
        # Process each row, predict, and store results
        predictions = []
        for index, row in df.iterrows():
            input_data = np.array(row[FEATURES]).reshape(1, -1)
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)
            predictions.append(prediction[0][0])
        
        # Add predictions to the DataFrame
        df["Predicted Power_kW"] = predictions
        
        # Save predictions to a new CSV file
        df.to_csv(output_path, index=False)
        
        return {
            "file_path": output_path,
            "success": True,
            "message": f"Successfully processed {len(predictions)} rows and saved to {output_path}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV processing error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)