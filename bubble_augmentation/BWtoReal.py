import tensorflow as tf
from tensorflow.keras import layers, models, utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

batch_size = 8
train_datagen = ImageDataGenerator(rescale=1./255, data_format='channels_last')
train_generator = train_datagen.flow_from_directory(
    "/home/iec/Documents/bubble_project/BubbleProject/datasets",
    target_size=(256, 256, 1),
    batch_size=batch_size,
    class_mode='input'
    )

train_ds = tf.keras.utils.image_dataset_from_directory(
  "/home/iec/Documents/bubble_project/BubbleProject/datasets",
  validation_split=0.2,
  color_mode='grayscale',
  subset="training",
  seed=123,
  label_mode=None,
  # shuffle=False,
  image_size=(256, 256),
  batch_size=64)

val_ds = tf.keras.utils.image_dataset_from_directory(
  "/home/iec/Documents/bubble_project/BubbleProject/datasets",
  validation_split=0.2,
  color_mode='grayscale',
  subset="validation",
  seed=123,
  label_mode=None,
  # shuffle=False,
  image_size=(256, 256),
  batch_size=64)

print(train_ds.take(2))

# create model
autoE = models.Sequential([
  #(256, 256, 1)
  layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 1)),
  layers.MaxPooling2D((2, 2), padding='same'),
  #(128,128,32)

  layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
  layers.MaxPooling2D((2, 2), padding='same'),
  #(64, 64, 32)

  layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
  layers.MaxPooling2D((2, 2), padding='same'),
  #(32, 32, 32)

  layers.Flatten(),
  layers.Dense(1024, activation='relu'),
  layers.Dense(512, activation='relu'),
  layers.Dense(1024, activation='relu'),
  # (1024, 1)

  layers.Reshape(target_shape=(32, 32, 1)),
  # (32, 32, 1)

  layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
  layers.UpSampling2D((2, 2)),
  # (64, 64, 32)

  layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
  layers.UpSampling2D((2, 2)),
  # (128, 128, 32)

  layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
  layers.UpSampling2D((2, 2)),
  # (256, 256, 32)

  layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
  # (256, 256, 1)
])

autoE.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))

autoE.fit_generator(train_generator, train_ds, epochs=10)

pred = autoE.predict(train_ds)

temp = pred[0]

min_val = np.min(temp)
max_val = np.max(temp)

# Rescale the values to the range 0-255
rescaled_data = (temp - min_val) * (255.0 / (max_val - min_val))

# Convert the rescaled array to integers
rescaled_data = rescaled_data.astype(np.uint8)

#print(rescaled_data)

# Create an image from the rescaled array
img = Image.fromarray(np.squeeze(rescaled_data))

img.save("/home/iec/Documents/bubble_project/BubbleProject/bubble_augmentation/runs/test.jpeg")