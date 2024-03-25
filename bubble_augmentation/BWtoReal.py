import tensorflow as tf
from tensorflow.keras import datasets, layers, models, utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

train_ds = tf.keras.utils.image_dataset_from_directory(
  "/home/iec/OneDrive/Ono Project/JAEA/Project_Files/autoencoderData",
  validation_split=0.2,
  color_mode='grayscale',
  subset="training",
  # seed=123,
  shuffle=False,
  image_size=(256, 256),
  batch_size=64)

val_ds = tf.keras.utils.image_dataset_from_directory(
  "/home/iec/OneDrive/Ono Project/JAEA/Project_Files/autoencoderData",
  validation_split=0.2,
  color_mode='grayscale',
  subset="validation",
  # seed=123,
  shuffle=False,
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

    # Encoder layers
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 1)),
    #(256, 256, 32)
    layers.MaxPooling2D((2, 2), padding='same'),
    #(128,128,32)
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    #(128, 128, 64)
    layers.MaxPooling2D((2, 2), padding='same'),
    #(64, 64, 64)
    
    # Decoder layers
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    # (64, 64, 64)
    layers.UpSampling2D((2, 2)),
    # (128, 128, 64)
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    # (128, 128, 32)
    layers.UpSampling2D((2, 2)),
    # (256, 256, 32)
    layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
    # (256, 256, 1)
    
])

autoE.compile(loss='mean_squared_error', optimizer='adam')

autoE.fit(x=train_ds, epochs=10, batch_size=64)

pred = autoE.predict(train_ds)