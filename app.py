import flask
import io
import numpy as np
from flask import Flask, jsonify, request
import keras
from PIL import Image
model = keras.models.load_model('./best_model.h5')

def prepare_image(img):
    img = Image.open(io.BytesIO(img))
    img = img.resize((256, 256))
    img = np.array(img)
    img = np.expand_dims(img, 0)
    return img

def predict_result(img):
    pred = np.argmax(model.predict(img))
    return str(pred)

app = Flask(__name__)

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

@app.route('/', methods=['GET'])
def index():
    return 'Machine Learning Inference'


# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0')