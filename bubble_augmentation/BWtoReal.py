import tensorflow as tf
from tensorflow.keras import datasets, layers, models, utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

train_ds = tf.keras.utils.image_dataset_from_directory(
  "/home/iec/OneDrive/Ono Project/JAEA/Project_Files/autoencoderData",
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(256, 256),
  batch_size=64)

val_ds = tf.keras.utils.image_dataset_from_directory(
  "/home/iec/OneDrive/Ono Project/JAEA/Project_Files/autoencoderData",
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(256, 256),
  batch_size=64)

# testImg = Image.fromarray(train_ds.take(1))
# Image.show(testImg)

# print(train_ds.class_names)

# Possibly find way to visualize data.

#Reshaping into 2D unraveled array and normalizing data
# X_train = X_train.reshape(X_train.shape[0], 784).astype('float32')
# X_test = X_test.reshape(X_test.shape[0], 784).astype('float32')

# X_train /= 255
# X_test /= 255

# create model
autoE = models.Sequential([
    layers.Dense(300, activation='relu'),
    layers.Dense(200, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(200, activation='relu'),
    layers.Dense(300, activation='relu'),
    layers.Dense(784, activation='sigmoid'),
])

autoE.compile(loss='mean_squared_error', optimizer='adam')

autoE.fit(train_ds, validation_data=val_ds, epochs=10, batch_size=1)

pred = autoE.predict(train_ds)