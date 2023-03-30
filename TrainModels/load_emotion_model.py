import tensorflow as tf
import os
import pathlib
# Import Libraries
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import os
import glob
import matplotlib.pyplot as plt
# Keras API
import keras
# Keras Model Imports
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation, AveragePooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt


# Save entire model with optimizer, architecture, weights and training configuration.
model = tf.keras.models.load_model('../model')
print("Model Loaded Successfully!!!")

# Save model weights.
from keras.models import load_model
model.save_weights('../model')
print("Weights Saved Successfully!!!")


# Loading model and predict.
from keras.models import load_model
model=load_model('../model')
Classes = ["angry","disgust","fear","happy","neutral","sad","surprise"]

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

# Pre-Processing test data same as train data.
img_width = 256
img_height = 256
# model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

from keras.preprocessing import image


def prepare(img_path):
    img = load_img(img_path, target_size=(256, 256))
    x = img_to_array(img)
    x = x / 255
    return np.expand_dims(x, axis=0)


# result = model.predict_classes([prepare('drive/My Drive/Emotions/surprise/35863.jpg')])
result = model.predict([prepare('../images/train/angry/10.jpg')])
classes_x = np.argmax(result, axis=1)
# result = np.argmax(model.predict([prepare('drive/My Drive/Emotions/surprise/35863.jpg')]),axis=1)
disease = load_img('../images/train/angry/10.jpg')
plt.imshow(disease)
print(Classes[int(classes_x)])