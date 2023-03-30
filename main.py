import flask
import io
import string
import time
import os
import numpy as np
import tensorflow as tf
from PIL import Image
# Keras API
import keras
from flask import Flask, jsonify, request
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('./model')
print("Model Loaded Successfully!!!")


# Loading model and predict.
from keras.models import load_model
model=load_model('./model')
Classes = ["angry","disgust","fear","happy","neutral","sad","surprise"]
def prepare_image(img):
    img = Image.open(io.BytesIO(img))
    img = img.resize((256, 256))
    img = np.array(img)
    img = np.expand_dims(img, 0)
    return img


img_width = 256
img_height = 256
def prepare(img):
    img_sec = load_img(img, target_size=(256, 256))
    x = img_to_array(img_sec)
    x = x / 255
    return np.expand_dims(x, axis=0)


def predict_result(img):
    result = model.predict(img)
    classes_x = np.argmax(result, axis=1)
    # result = np.argmax(model.predict([prepare('drive/My Drive/Emotions/surprise/35863.jpg')]),axis=1)
    """emotion = load_img(img)
    plt.imshow(emotion)
    print(emotion)"""
    return (Classes[int(classes_x)])


app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def infer_image():
    if 'file' not in request.files:
        return "Please try again. The Image doesn't exist"

    file = request.files.get('file')

    if not file:
        return

    """img_bytes = file.read()
    img = prepare(img_path=)"""
#    img = prepare_image()

    img_bytes = file.read()
    img = prepare_image(img_bytes)

    return jsonify(prediction=predict_result(img))


@app.route('/', methods=['GET'])
def index():
    return 'Machine Learning Inference'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')