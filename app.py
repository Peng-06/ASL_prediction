from flask import Flask, jsonify, request, send_file
import os
import numpy as np
from PIL import Image
import keras
import io

app = Flask(__name__) #__name__ = __main__ -----> APP
print('Loading model')
model = keras.models.load_model('ASL.h5')

input_shape = model.input_shape
IMG_SIZE = input_shape[1]

classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O',
           'P','Q','R','S','T','U','V','W','X','Y','Z','del','nothing','space']

@app.route('/')
def home():
    return send_file("./index.html")

@app.route("/predict",methods = ["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error" : "No image provided"}),400 #bad request

    
    file = request.files["image"]
    img = Image.open(io.BytesIO(file.read()))
    img = img.resize((IMG_SIZE,IMG_SIZE),Image.Resampling.BILINEAR)
    img = img.convert('RGB')
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis = 0)

    prediction = model.predict(img_array, verbose = 0)
    predicted_class = classes[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    return jsonify({
        "prediction" : predicted_class,
        "confidence" : round(confidence,4)
    })


if __name__ == "__main__":
    app.run(port=5000)
