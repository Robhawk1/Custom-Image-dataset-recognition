# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 15:21:04 2018

@author: Rob_hawk
"""

import os

#Creating dataset path
path = 'C:\\Users\\User\\Desktop\\jutsu_api\\dataset'
train_dir = os.path.join(path,'train')
test_dir = os.path.join(path,'test')
num_classes = len(os.listdir(train_dir))

#from keras.optimizers import SGD
#from sklearn.preprocessing import LabelBinarizer

#Importing VGG16 model with weights
from keras.applications import VGG16
conv_base = VGG16(weights = 'imagenet',
                  include_top = False,
                  input_shape=(480,640,3))

from keras import models
from keras import layers


model = models.Sequential()
model.add(conv_base)
model.summary()
conv_base.summary()

#Adding layers
model.add(layers.Flatten())
model.add(layers.Dense(512, activation = 'relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation = 'softmax'))
model.summary()

conv_base.trainable = False
"""
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
"""        
 
#Importing data generator       
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

train_datagen = ImageDataGenerator(
        rescale = 1./255,
        rotation_range = 40,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        vertical_flip = True,
        fill_mode = 'nearest')        
        
test_datagen = ImageDataGenerator(rescale = 1./255)
batch_size = 2
        
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size = (480,640),
        batch_size = batch_size,
        class_mode = 'categorical')

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size = (480,640),
        batch_size = batch_size,
        class_mode = 'categorical')

model.compile(loss='categorical_crossentropy',
              optimizer = optimizers.Adam(lr=.001),
              metrics = ['categorical_accuracy'])

#fitting the model    
history = model.fit_generator(
        train_generator,
        steps_per_epoch = 50,
        epochs = 25,
        validation_data = test_generator,
        validation_steps = 50)


acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc) +1)
import matplotlib.pyplot as plt
plt.plot(epochs, acc, 'bo', label = 'Training acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

        
        
        
        