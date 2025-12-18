import tensorflow as tf
import os

MODEL_PATH = "vision_data/fittonia_final_model.keras"
OUTPUT_PATH = "vision_data/fittonia_final_model.tflite"

def convert():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    print("Loading Keras model...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    with open(OUTPUT_PATH, "wb") as f:
        f.write(tflite_model)

    print(f"Saved TFLite model → {OUTPUT_PATH}")

if __name__ == "__main__":
    convert()
