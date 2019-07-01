# CNN for AD classification
This repository contains all code used in the research project applying deep learning for the classification of Alzheimer's Disease (AD).  

## nii_to_npy.py
This script converts all nifti MRI data into numpy arrays. This way the data is applicable to a Keras data generator, which controls the input data of the deep learning model. Besides conversion to npy, this script applies a mask of the brain to the MRI data and sets all outside values to zero. Furthermore, based on this mask the images are cropped. As extra option the images can be down sampled as well.

## config.py
This is the configuration file which contains all deep learning settings for AD classification. For example the task, input data type, pre-training and augmentation options can be adjusted or the amount of cross validations, epochs and batch size can be entered. This file is imported in all following scripts, so that its variables can be called.  

## main.py
Running this script will manage the training and evaluation of the deep learning model. The main script will call all other functions which are used for splitting the data, augmentation, normalization etc. This script will create a output directory in which the results of every cross-validation will be stored, together with the best models and the mean and std used for normalization. Per fold plots will be provided about the train and validate loss, acc, AUC, sensitivity and specificity per epoch. Furthermore, a plot will be provided of the test AUC of all folds. The train-val-test subjects, configuration file and model summary are stored in the output directory as well. The results of all folds are stored in a dictionary named 'results.npy'. 




 