import tensorflow as tf
import keras

model=keras.models.load_model("ASL.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("ASL.tflite","wb") as f:
    f.write(tflite_model)