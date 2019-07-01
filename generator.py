# import configuration file
import config

# set random seed
from numpy.random import seed
from tensorflow import set_random_seed
seed(config.fixed_seed)
set_random_seed(config.fixed_seed)

import numpy as np
import keras


class DataGenerator(keras.utils.Sequence):
    """
    Generates data for Keras
    Code is adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    All images are loaded and normalized based on the given mean and std of the train set.
    """
    def __init__(self, list_IDs, labels, mean, std, batch_size=16, dim=(32,32,32), n_channels=1,
                 n_classes=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.list_IDs_temp = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.mean = mean
        self.std = std
        self.data_dir = config.data_dir
        self.aug_dir = config.aug_dir
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        # samples / batch size
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))

        # shuffle order of input examples
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # Store sample: load original and augmented images
            if ID[0] != 'a':
                X[i,] = np.reshape(np.load(self.data_dir + ID + '.npy'), (self.dim[0], self.dim[1], self.dim[2], self.n_channels))
            else:
                X[i,] = np.reshape(np.load(self.aug_dir + ID + '.npy'), (self.dim[0], self.dim[1], self.dim[2], self.n_channels))

            # normalization
            X[i,] = np.subtract(X[i,], np.reshape(self.mean, (self.dim[0], self.dim[1], self.dim[2], self.n_channels)))
            X[i,] = np.divide(X[i,], np.reshape(self.std, (self.dim[0], self.dim[1], self.dim[2], self.n_channels)))

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

