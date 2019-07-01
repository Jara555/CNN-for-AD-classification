# import configuration file
import config

# set random seed
from numpy.random import seed
from tensorflow import set_random_seed
seed(config.fixed_seed)
set_random_seed(config.fixed_seed)

import numpy as np


def standardization_matrix(dataset):
    """
    Returns the voxel-wise mean and std of an input dataset
    Note: the standardization matrix computed from the training data should also be
    used for the validation and test data.

        INPUT:
            dataset - the dataset containing all IDs of the subjects which should be used for normalization

        OUTPUT:
            mean - 3D matrix containing the voxel-wise mean computed from all subjects of the input dataset
            std - 3D matrix containing the voxel-wise std computed from all subjects of the input dataset
    """

    print("\nCompute standardization matrix")
    mean = mean_matrix(dataset)
    print("    mean : computed")
    std = std_matrix(mean, dataset)
    print("    std : computed\n")

    return mean, std


def mean_matrix(dataset):
    """
    Returns the voxel-wise mean of an input dataset
    """

    print("    mean : in progress")

    # loop over every input file and calculate mean
    mean = np.zeros(config.input_shape)
    for id in dataset:
        if id[0] != 'a':
            x = np.load(config.data_dir + id + '.npy')
        else:
            x = np.load(config.aug_dir + id + '.npy')
        mean = mean + x / len(dataset)

    return mean


def std_matrix(mean, dataset):
    """
    Returns the voxel-wise std of an input dataset based on the mean of that dataset
    """

    print("    std : in progress")

    # loop over every input file and calculate std
    std = np.zeros(config.input_shape)
    for id in dataset:
        if id[0] != 'a':
            x = np.load(config.data_dir + id + '.npy')
        else:
            x = np.load(config.aug_dir + id + '.npy')
        std = std + np.square(x - mean) / len(dataset)

    std = np.sqrt(std)

    # to avoid dividing by 0 add 1e-20 to the std
    std = std + np.ones(config.input_shape) * 1e-20

    return std

