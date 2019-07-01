# CNN for AD classification
This repository contains all code used in the research project Deep Learning for the classification of Alzheimer's Disease.  

## nii_to_npy.py
This script converts all nifti MRI data into numpy arrays. This way the data can be used by a Keras data generator, which controls the data feeded to the deep learning model. Besides conversion to npy, this script applies a mask of the brain to the MRI data and sets all outside values to zero. As extra option down sampled versions of the MR images can be created as well.

## config.py
This is the configuration file which contains all deep learning settings for AD classification. For example the task, input data type, pre-training and augmentation options can be adjusted or the amount of cross validations, epochs and batch size can be entered. This variables specified in this configuration file are imported in all following scripts required for the implementation of the deep learning network.  

## main.py
Running this script will manage the training and evaluation of the deep learning model. The main script will call all other functions which are used for splitting the data, augmentation, normalization etc. This script will create a output directory in which the results of every cross-validation will be stored, together with the best models and the mean and std used for normalization. Per fold plots will be provided demonstrating the train and validate loss, acc, AUC, sensitivity and specificity per epoch. Furthermore, a plot will be provided of the test ROC-AUC of all folds. The train/val/test subject splits, configuration file and model summary are stored in the output directory as well. The results of all folds are stored in a dictionary named _results.npy_ from which results can be accessed like _results["train/test/val"]["acc/loss/AUC"]["k"]_.

## MCI_crossval.py
This script is an alternative version of 'main.py' which has the purpose of running a complete data set through a pre-trained model as evaluation only. This can be used for MCI classification, since for this task it is beneficial to use a network which is pre-trained on the AD task. For every model created in each fold of the cross-validation of the AD task, all MCI data is used to evaluate this model. In this way the MCI results of each fold can be averaged for the final performance. Similar as for the _main.py_ script this script uses the configuration settings of the _config.py_ file. Also this script saves all evaluation results in a _results.npy_ dictionary and provides a plot of the ROC-AUC of all folds. The configuration file and model information will be saved.

## stats.py
This script can be run to perform a statistical McNemar test comparing the correct classifications of two methods. For this the y_pred and y_true values are required which are automatically saved in every fold of the cross-validation when running _main.py_.

# Visualization
The scripts which can be found in the visualization folder implement a gradCAM analysis of the AD classification model. In the configuration file _vis_config.py_ all settings required for this analysis can be specified. First of all the _get_missclassifications.py_ script should be run to obtain the correct and miss classified subjects per class. These are required for computation of the Grad-CAM images in the following scripts. Since in most cases computational memory is restricted to calculating the Grad-CAM of only 10 subjects in one run, the subjects will be split in batches of 10. These batches will be used in the _vis_main.py_ script to calculate an average Grad-CAM. For this reason the _vis_main.py_ needs a integer defining the run as input argument. In that way the correct batch with subjects can be selected to process. After running this script the average Grad-CAMs created in each run can be calculated using the script _vis_average.py_. Here a final nifti version in MNI152 format of both the Grad-CAM mean and variation will be created which can be used for visualization. The gradcam computations are adapted from the code on: https://github.com/eclique/keras-gradcam/blob/master/grad_cam.py and https://github.com/keras-team/keras/blob/master/examples/conv_filter_visualization.py Implementation is based on the paper of Selvaraju et al. (2017): https://arxiv.org/pdf/1610.02391.pdf

# Author
- Jara Linders 
- Biomedical Imaging Group Rotterdam (BIGR)




 
