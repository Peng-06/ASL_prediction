from flask import Flask, jsonify, request, send_file
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import io

app = Flask(__name__) #__name__ = __main__ -----> APP
print('Loading model')
interpreter = tf.lite.Interpreter('ASL.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


IMG_SIZE = input_details[0]['shape'][1]

classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O',
           'P','Q','R','S','T','U','V','W','X','Y','Z','del','nothing','space']

@app.route('/')
def home():
    return send_file("./index.html")

@app.route("/predict",methods = ["POST"])
def predict():
    try:

        if 'image' not in request.files:
            return jsonify({"error" : "No image provided"}),400 #bad request

        
        file = request.files["image"]
        img = Image.open(io.BytesIO(file.read()))
        img = img.resize((IMG_SIZE,IMG_SIZE),Image.Resampling.BILINEAR)
        img = img.convert('RGB')
        img_array = np.array(img)/255.0
        img_array = np.expand_dims(img_array, axis = 0).astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'],img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = classes[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        return jsonify({
            "prediction" : predicted_class,
            "confidence" : round(confidence,4)
        })
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}),500

if __name__ == "__main__":
    port = int(os.environ.get('PORT',5000))
    app.run(host='0.0.0.0',port=port)
