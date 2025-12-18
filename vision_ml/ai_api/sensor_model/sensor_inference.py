import numpy as np
import joblib
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
from pathlib import Path

from .light_rules import compute_too_bright_now, FITTONIA_PROFILE

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "sensor_data" / "sensor_multilabel_model.tflite"
SCALER_PATH = BASE_DIR / "sensor_data" / "sensor_scaler.save"

FEATURES = [
    "light",
    "temperature",
    "soil_capacitance",
    "hours_since_last_watering",
    "delta_soil",
    "rolling_avg_light_6h",
    "light_hours_today",
    "hour_of_day",
]

LABELS = [
    "forgot_to_water",
    "too_dark_today",
    "worth_relocating",
]

THRESHOLDS = {
    "forgot_to_water": 0.5,
    "too_dark_today": 0.5,
    "worth_relocating": 0.5,
}

scaler = joblib.load(SCALER_PATH)

interpreter = Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def _model_predict_proba(sample_dict: dict) -> np.ndarray:
    """
    Przyjmuje sample_dict z FEATURE -> wartość
    Zwraca np.array shape (3,) z prawdopodobieństwami dla etykiet.
    """
    x = np.array([[sample_dict[f] for f in FEATURES]], dtype=np.float32)  # shape (1, 8)

    x_scaled = scaler.transform(x).astype(np.float32)

    interpreter.set_tensor(input_details[0]["index"], x_scaled)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])  # shape (1, 3)

    return output_data[0]  # (3,)


def predict_sensor(sample_dict: dict) -> dict:
    probs = _model_predict_proba(sample_dict)

    flags = {}
    probas = {}

    for i, label in enumerate(LABELS):
        p = float(probs[i])
        probas[label] = p
        flags[label] = bool(p >= THRESHOLDS[label])

    too_bright = compute_too_bright_now(
        light=sample_dict["light"],
        rolling_avg_light_6h=sample_dict["rolling_avg_light_6h"],
        light_hours_today=sample_dict["light_hours_today"],
        profile=FITTONIA_PROFILE
    )

    flags["too_bright_now"] = bool(too_bright)
    probas["too_bright_now"] = None  # to reguła nie ML

    return {
        "flags": flags,
        "probas": probas,
        "thresholds": THRESHOLDS
    }

