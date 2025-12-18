import uvicorn

from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from .vision_model.vision_inference import predict_plant_state
from fastapi.middleware.cors import CORSMiddleware
from .data.influx_reader import build_sensor_input
from .sensor_model.sensor_inference import predict_sensor
import json


app = FastAPI(
    title="Plant AI API",
    description="Vision + Sensor AI for Fittonia health monitoring",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SensorInput(BaseModel):
    light: float
    temperature: float
    soil_capacitance: float
    hours_since_last_watering: float
    delta_soil: float
    rolling_avg_light_6h: float
    light_hours_today: float
    hour_of_day: int


@app.post("/sensor/predict")
def predict_sensor_state(data: SensorInput):
    """
    Predict labels from sensor model + rule-based logic.
    """
    result = predict_sensor(data.model_dump())
    return {
        "sensor_AI_result": result
    }

@app.post("/sensor/predict/live")
def predict_sensor_live():
    sample = build_sensor_input()
    result = predict_sensor(sample)
    return {
        "source": "influxdb",
        "sensor_AI_result": result,
        "features": sample
    }

@app.post("/vision/predict")
async def predict_vision(file: UploadFile = File(...)):
    """Upload plant image → get classification result."""
    contents = await file.read()
    result = predict_plant_state(contents)
    return {"vision_AI_result": result}

@app.post("/combined/predict")
async def combined_predict(
    data: str = Form(...),
    file: UploadFile = File(...)
):
    sensor_dict = json.loads(data)
    sensor_out = predict_sensor(sensor_dict)
    vision_out = predict_plant_state(await file.read())
    return {"sensor": sensor_out, "vision": vision_out}

@app.post("/combined/predict/live")
async def combined_predict_live(file: UploadFile = File(...)):
    """
    Live combined prediction:
    - sensors are read from InfluxDB
    - image is provided by client
    """
    sample = build_sensor_input()
    sensor_out = predict_sensor(sample)

    img_bytes = await file.read()
    vision_out = predict_plant_state(img_bytes)

    return {
        "source": "influxdb + image",
        "sensor": sensor_out,
        "vision": vision_out,
        "features": sample
    }

@app.get("/")
def root():
    return {
        "status": "Plant AI API running",
        "endpoints": [
            "/sensor/predict",
            "/vision/predict",
            "/combined/predict"
        ]
    }

if __name__ == "__main__":
    # ważne: pełna ścieżka modułu, nie "main:app"
    uvicorn.run("ai_api.main:app", host="0.0.0.0", port=8000, reload=True)
