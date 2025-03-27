from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Load the trained ML model
model = joblib.load("battery_model.pkl")

@app.get("/")
def home():
    return {"message": "EV Fleet Monitoring API is running!"}

@app.get("/predict_battery_failure/")
def predict_battery_failure(battery_health: float, charging_cycles: int, temperature: float):
    prediction = model.predict(np.array([[battery_health, charging_cycles, temperature]]))
    return {"failure_risk": int(prediction[0])}
