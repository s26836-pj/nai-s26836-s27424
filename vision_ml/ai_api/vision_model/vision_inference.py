import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
VISION_DATA_DIR = BASE_DIR / "vision_data"
VISION_MODEL_PATH = VISION_DATA_DIR / "fittonia_final_model.tflite"
CLASS_INDICES_PATH = VISION_DATA_DIR / "class_indices.json"

with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
    idx_to_class = {int(k): v for k, v in data["idx_to_class"].items()}

VISION_LABELS = [idx_to_class[i] for i in range(len(idx_to_class))]


interpreter = Interpreter(model_path=str(VISION_MODEL_PATH))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

INPUT_INDEX = input_details[0]["index"]
OUTPUT_INDEX = output_details[0]["index"]

_, height, width, _ = input_details[0]["shape"]


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Przyjmuje raw bytes z UploadFile i zwraca tablicę float32 [1, H, W, 3]
    w zakresie [0, 1].
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((width, height))

    arr = np.array(img).astype(np.float32)
    arr = (arr / 127.5) - 1.0  # MobileNetV2 preprocess
    arr = np.expand_dims(arr, axis=0)
    return arr


def _aggregate_bucket(label: str) -> str:
    """
    Prostą klasyfikację na 'healthy' / 'unhealthy' można zrobić z sufiksów.
    """
    if label.startswith("healthy_"):
        return "healthy"
    if label.startswith("unhealthy_"):
        return "unhealthy"
    return "unknown"


def predict_plant_state(image_bytes: bytes) -> dict:
    """
    Zwraca dict:
      {
        "healthy_low": p0,
        "healthy_mid": p1,
        ...
        "top_label": ...,
        "top_score": ...,
        "top_bucket": "healthy" / "unhealthy"
      }
    """
    x = preprocess_image(image_bytes)

    interpreter.set_tensor(INPUT_INDEX, x)
    interpreter.invoke()

    output = interpreter.get_tensor(OUTPUT_INDEX)  # [1, n_classes]
    logits = output[0]

    # softmax na wszelki wypadek jesli model tflite nie ma aktywacji
    exps = np.exp(logits - np.max(logits))
    softmax = exps / np.sum(exps)

    result: dict = {}
    for label, p in zip(VISION_LABELS, softmax):
        result[label] = float(p)

    # najlepsza klasa
    top_idx = int(np.argmax(softmax))
    top_label = VISION_LABELS[top_idx]
    top_score = float(softmax[top_idx])

    result["top_label"] = top_label
    result["top_score"] = top_score
    result["top_bucket"] = _aggregate_bucket(top_label)

    return result
