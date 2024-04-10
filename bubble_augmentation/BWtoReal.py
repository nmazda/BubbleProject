import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import Sequence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os
from PIL import Image
from sklearn.model_selection import train_test_split

tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

data_dir = '/home/iec/Documents/bubble_project/BubbleProject/datasets/'
bw_dir = '/home/iec/Documents/bubble_project/BubbleProject/datasets/blackAndWhite/'
real_dir = '/home/iec/Documents/bubble_project/BubbleProject/datasets/real/'

class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, list_IDs, labels, image_path, mask_path,
                 to_fit=True, batch_size=32, dim=(256, 256),
                 n_channels=1, n_classes=10, shuffle=True):
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param labels: list of image labels (file names)
        :param image_path: path to images location
        :param mask_path: path to masks location
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.list_IDs = list_IDs
        self.labels = labels
        self.image_path = image_path
        self.mask_path = mask_path
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self._generate_X(list_IDs_temp)

        if self.to_fit:
            y = self._generate_y(list_IDs_temp)
            return X, y
        else:
            return X

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _generate_X(self, list_IDs_temp):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,:,:,0] = self._load_grayscale_image(self.image_path + self.labels[ID])

        return X

    def _generate_y(self, list_IDs_temp):
        """Generates data containing batch_size masks
        :param list_IDs_temp: list of label ids to load
        :return: batch if masks
        """
        y = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            y[i,:,:, 0] = self._load_grayscale_image(self.mask_path + self.labels[ID])

        return y

    def _load_grayscale_image(self, image_path):
        """Load grayscale image
        :param image_path: path to image to load
        :return: loaded image
        """
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img / 255
        return img
    

fileNames = os.listdir(real_dir)

# Split data into train and validation sets
train_files, val_files = train_test_split(fileNames, test_size=0.2, random_state=42)

# Assign unique IDs to train and validation data
train_ids = list(range(len(train_files)))
val_ids = list(range(len(val_files)))

# Create DataGenerators for train and validation data
train_generator = DataGenerator(labels=train_files, list_IDs=train_ids, image_path=bw_dir, mask_path=real_dir, batch_size=32)
val_generator = DataGenerator(labels=val_files, list_IDs=val_ids, image_path=bw_dir, mask_path=real_dir, batch_size=32)
    
X_batch, y_batch = train_generator.__getitem__(0)

x_example = X_batch[0, :, :, 0] * 255
y_example = y_batch[0, :, :, 0] * 255

x_arr = np.squeeze(x_example)
y_arr = np.squeeze(y_example)

# Ensure x_arr and y_arr are within the range [0, 255]
x_arr = np.clip(x_arr, 0, 255)
y_arr = np.clip(y_arr, 0, 255)

# Normalize pixel values to the range [0, 255]
x_norm = np.interp(x_arr, (x_arr.min(), x_arr.max()), (0, 255)).astype(np.uint8)
y_norm = np.interp(y_arr, (y_arr.min(), y_arr.max()), (0, 255)).astype(np.uint8)

# Create Pillow Image objects from the normalized arrays
x_img = Image.fromarray(x_norm)
y_img = Image.fromarray(y_norm)

# Save X and Y images
x_img.save("/home/iec/Documents/bubble_project/BubbleProject/bubble_augmentation/runs/X.jpeg")
y_img.save("/home/iec/Documents/bubble_project/BubbleProject/bubble_augmentation/runs/Y.jpeg")

# create model
autoE = models.Sequential([
  #(256, 256, 1)
  layers.Conv2D(32, kernel_size=(3, 3), strides=(2,2), activation='leaky_relu', padding='same', input_shape=(256, 256, 1)),
  layers.Conv2D(32, kernel_size=(3, 3), strides=(2,2), activation='leaky_relu', padding='same'),
  layers.Conv2D(32, kernel_size=(3, 3), strides=(2,2), activation='leaky_relu', padding='same'),
  #(32, 32, 32)

  layers.Flatten(),
  layers.Dense(4096, activation='leaky_relu'),
  layers.Dense(4096, activation='leaky_relu'),
  layers.Dense(4096, activation='leaky_relu'),
  layers.Reshape((32, 32, 4)),

  #(32, 32, 32)
  layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
  layers.UpSampling2D((2, 2)),
  layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
  layers.UpSampling2D((2, 2)),
  layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
  layers.UpSampling2D((2, 2)),
  # (256, 256, 32)



  layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')
  # (256, 256, 1)
])

autoE.summary()

#Early stopping, not sure if needs to be implemented just yet.
#early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# mean_absolute_error or mean_squared_error
autoE.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))

history = autoE.fit(train_generator, validation_data=val_generator, epochs=20)



#Get and predict a batch of images
image = val_generator.__getitem__(0)
pred = autoE.predict(image[0])

#Printing shape to ensure right size.
print(image[0].shape)
print(image[1].shape)
print(pred.shape)

#Get the individual images
arrPred = pred[0]
arrX = image[0][0]
arrY = image[1][0]

# Rescale the values to the range 0-255
rescaled_dataPred = np.interp(arrPred, (arrPred.min(), arrPred.max()), (0, 255)).astype(np.uint8)
rescaled_dataX = np.interp(arrX, (arrX.min(), arrX.max()), (0, 255)).astype(np.uint8)
rescaled_dataY = np.interp(arrY, (arrY.min(), arrY.max()), (0, 255)).astype(np.uint8)

# Create an image from the rescaled array
imgPred = Image.fromarray(np.squeeze(rescaled_dataPred))
imgX = Image.fromarray(np.squeeze(rescaled_dataX))
imgY = Image.fromarray(np.squeeze(rescaled_dataY))

#Saving images to testPred, testX, testY
imgPred.save("/home/iec/Documents/bubble_project/BubbleProject/bubble_augmentation/runs/testPred.jpeg")
imgX.save("/home/iec/Documents/bubble_project/BubbleProject/bubble_augmentation/runs/testX.jpeg")
imgY.save("/home/iec/Documents/bubble_project/BubbleProject/bubble_augmentation/runs/testY.jpeg")

#Plotting loss curve and saving to loss_curve.jpeg
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('AE model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('/home/iec/Documents/bubble_project/BubbleProject/bubble_augmentation/runs/loss_curve.jpeg')