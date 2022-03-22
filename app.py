import flask
import io
import numpy as np
from flask import Flask, jsonify, request , send_from_directory
import keras
from PIL import Image
from flask_cors import CORS, cross_origin
model = keras.models.load_model('./best_model.h5')
import os
def prepare_image(img):
    img = Image.open(io.BytesIO(img))
    img = img.resize((256, 256))
    img = np.array(img)
    img = np.expand_dims(img, 0)
    return img

def predict_result(img):
    pred = np.argmax(model.predict(img))
    return str(pred)

app = Flask(__name__,static_folder="./build")
CORS(app)

@app.route('/predict', methods=['POST'])
def infer_image():
    if 'file' not in request.files:
        return "Please try again. The Image doesn't exist"
    
    file = request.files.get('file')

    if not file:
        return

    img_bytes = file.read()
    img = prepare_image(img_bytes)

    return jsonify(prediction=predict_result(img))

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0')