import tensorflow as tf
from tensorflow.keras import datasets, layers, models, utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)

(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

X_train = X_train[:600]
y_train = y_train[:600]

X_test = X_test[:100]
y_test = y_test[:100]

print(" ")

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# Possibly find way to visualize data.

#Reshaping into 2D unraveled array and normalizing data
X_train = X_train.reshape(X_train.shape[0], 784).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 784).astype('float32')

X_train /= 255
X_test /= 255

print(X_train.shape)
print(X_test.shape)

#Add noise to array
noise_factor = 0.2
X_train_noisy = X_train + (noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape))
X_test_noisy = X_test + (noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape))

#If noise goes above 1, clip to 1
X_train_noisy = np.clip(X_train_noisy, 0., 1.)
X_test_noisy = np.clip(X_test_noisy, 0., 1.)

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

autoE.fit(X_train_noisy, X_train, validation_data=(X_test_noisy, X_test), epochs=10, batch_size=1)

pred = autoE.predict(X_test_noisy)