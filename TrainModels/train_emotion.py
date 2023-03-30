import tensorflow as tf
import os
import pathlib
# Import Libraries
import warnings
warnings.filterwarnings("ignore")
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

print("Package Import Success!!!")

# Fetch Data Set From Local
train_dir = "../images/train"
test_dir = "../images/validation"

print("Get Images Success!!!")

# function to get count of images
def get_files(directory):
  if not os.path.exists(directory):
    return 0
  count=0
  for current_path,dirs,files in os.walk(directory):
    for dr in dirs:
      count+= len(glob.glob(os.path.join(current_path,dr+"/*")))
  return count


train_samples =get_files(train_dir)
num_classes=len(glob.glob(train_dir+"/*"))
test_samples=get_files(test_dir) # For testing i took only few samples from unseen data. we can evaluate using validation data which is part of train data.
print(num_classes,"Classes")
print(train_samples,"Train images")
print(test_samples,"Test images")


train_datagen=ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   validation_split=0.2, # validation split 20%.
                                   horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)


# Preprocessing data.
train_datagen=ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   validation_split=0.2, # validation split 20%.
                                   horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)


print("Preprocessing Done!!!")


# set height and width and color of input image.
img_width,img_height =256,256
input_shape=(img_width,img_height,3)
#DONE: Batch Size Need to be reduced (1-5)
batch_size = 5

train_generator =train_datagen.flow_from_directory(train_dir,
                                                   target_size=(img_width,img_height),
                                                   batch_size=batch_size)
test_generator=test_datagen.flow_from_directory(test_dir,shuffle=True,
                                                   target_size=(img_width,img_height),
                                                   batch_size=batch_size)


print("Set Image Dimesion is Done!!!")

train_generator.class_indices


# CNN building.
model = Sequential()
model.add(Conv2D(32, (5, 5),input_shape=input_shape,activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(32, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128,activation='relu'))
model.add(Dense(num_classes,activation='softmax'))
model.summary()


#Print Model Layers

model_layers = [ layer.name for layer in model.layers]
print('layer name : ',model_layers)


from keras.preprocessing import image
import numpy as np
img1 = keras.utils.load_img('../images/train/angry/56.jpg')
plt.imshow(img1);

#preprocess image
img1 = keras.utils.load_img('../images/train/angry/56.jpg')
img = keras.utils.img_to_array(img1)
img = img/255


print("Image Preprocess is Done!!!")

# validation data.
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    # same directory as training data
    target_size=(img_height, img_width),
    batch_size=batch_size)

#Model building to get trained with parameters.
opt=keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
train=model.fit_generator(train_generator,
                          epochs=10,
                          steps_per_epoch=train_generator.samples//batch_size,
                          validation_data=validation_generator,
                          validation_steps=validation_generator.samples // batch_size,verbose=1)

print("Model Trained Successfully!!!!")

#os.makedirs('models/')

print("Directory Created")
tf.keras.models.save_model(model,'../model',)
tf.keras.models.save_model(model,'../model/emotion_model.h5')



accuracy = train.history['accuracy']
val_accuracy = train.history['val_accuracy']
loss = train.history['loss']
val_loss = train.history['val_loss']
epochs = range(1, len(accuracy) + 1)

#Train and validation accuracy
plt.plot(epochs, accuracy, 'b', label='Training accurarcy')
plt.plot(epochs, val_accuracy, 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()
plt.figure()

#Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()

