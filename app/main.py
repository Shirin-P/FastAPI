#Create FastAPI

# Import necessary libraries
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
# Initialize FastAPI app
#app = FastAPI()

# Load the trained .keras model
#model = joblib.load('D:\Crop Model\linear_regression_model.pkl')

# Define input data structure
#class PredictionInput(BaseModel):
#    Rainfall_mm: float
#    Temperature_Celsius: float
#    Days_to_Harvest: int
    # Add other fields here as per your model’s input features, such as Region, Soil_Type, etc.

# Prediction endpoint
#@app.get("/predict/")
#async def predict(Rainfall_mm: float, Temperature_Celsius: float, Days_to_Harvest: int):
#    try:
        # Convert input data to model's expected format
        #data = np.array([[input_data.Rainfall_mm, input_data.Temperature_Celsius, input_data.Days_to_Harvest]])
#        data = np.array([[Rainfall_mm, Temperature_Celsius, Days_to_Harvest]])

        # Make prediction
#        prediction = model.predict(data)

        # Return prediction
#        return {"predicted_yield": prediction[0]}

#    except Exception as e:
#        raise HTTPException(status_code=500, detail=str(e))



# Initialize FastAPI app
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Update with your frontend's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model (ensure this path is correct for your system)
model = joblib.load('D:/Crop Model/linear_regression_model.pkl')


@app.get("/predict/")
async def predict(
    feature_1: float,
    feature_2: float,
    feature_3: float,
    feature_4: float,
    feature_5: float,
    feature_6: float,
    feature_7: float,
    feature_8: float,
    feature_9: float,
    feature_10: float,
    feature_11: float,
    feature_12: float,
    feature_13: float,
    feature_14: float,
    feature_15: float,
    feature_16: float,
    feature_17: float,
    feature_18: float,
    feature_19: float,
    feature_20: float,
):
    # Verify features are numeric
    data = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10,
            feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20]

    if not all(isinstance(value, float) for value in data):
        raise HTTPException(status_code=422, detail="All features must be numeric and provided.")

    # Prediction logic
    prediction = model.predict([data])
    return {"predicted_yield": str(prediction[0])}