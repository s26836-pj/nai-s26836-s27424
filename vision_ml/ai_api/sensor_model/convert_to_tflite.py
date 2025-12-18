import tensorflow as tf

keras_model_path = "sensor_data/sensor_multilabel_model.keras"
tflite_model_path = "sensor_data/sensor_multilabel_model.tflite"

model = tf.keras.models.load_model(keras_model_path)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
#
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print("Zapisano:", tflite_model_path)
