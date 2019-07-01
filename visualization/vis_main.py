import datetime
import os
import sys
from shutil import copyfile

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.engine.saving import load_model
from skimage.transform import resize

# import configuration parameters
from visualization.vis_config import roi, task, label, classification_type, data_set, gc_layer, server, class_limit


def main():
    """
    The current script implements a gradCAM analysis of an AD classification model.
    In the configuration file 'vis_config.py' the settings can be specified.

    This script computes the Grad-CAM mean and variation of a batch of images. Since in most
    cases memory is restricted to calculating the Grad-CAM of only 10 subjects in one run, this script
    is designed to be run several times after each other until a Grad-CAM of all subjects is calculated.
    For this reason this script uses an input argument which defines the run, and can based on this select
    the 10 subjects which belong in that run.

    After running this script the Grad-CAMs created in each run can be calculated using the script
    'vis_average.py'. Here a final nifti version of both the mean and variation will be created.

    The gradcam computations are adapted from the code on:
    https://github.com/eclique/keras-gradcam/blob/master/grad_cam.py
    https://github.com/keras-team/keras/blob/master/examples/conv_filter_visualization.py

    Implementation is based on the paper of Selvaraju et al. (2017):
    https://arxiv.org/pdf/1610.02391.pdf
    """

    # set info paths
    if server:
        info_path = f"/server/path/to/gradcam/info/{roi}/{task}/info/"
    else:
        info_path = f"/local/path/to/gradcam/info/{roi}/{task}/info/"

    # load subject file
    all_subjects = np.load(f"{info_path}{classification_type}_classified_subjects_{label}.npy")
    k_splits = all_subjects.shape[0]

    # set paths (when running on server based on data dir + job nr input)
    if server:
        # run indicates which split to take
        run = int(sys.argv[3])
        data_path = f"/server/path/to/data/data_{roi}/"
        save_path = f"/save/path/for/gradcam/{roi}/{task}/{sys.argv[2]}_n{k_splits*class_limit}_{label}_c{gc_layer}_{classification_type}/"
        run_path = f"{save_path}{sys.argv[2]}_run{run}_{roi}_{data_set}_n{class_limit}_gc{gc_layer}_gb{gb_layer}_{label}_GradCAM/"
    else:
        run = 0
        data_path = f"/local/path/to/data/data_{roi}/"
        stamp = datetime.datetime.now().isoformat()[:16]
        save_path = f"/home/jara/gradcam/{roi}/{task}/{stamp}_n{class_limit*k_splits}_{label}_c{gc_layer}_{classification_type}/"
        run_path = f"{save_path}{roi}_{data_set}_n{class_limit}_gc{gc_layer}_gb{gb_layer}_{label}_GradCAM/"

    create_data_directory(run_path)
    if run == 0:
        if server:
            copyfile("/server/path/to/config/file/vis_config.py", f"{save_path}configuration.py")
        else:
            copyfile("/local/path/to/config/file/vis_config.py", f"{save_path}configuration.py")

    # set model information files
    mean_file = info_path + "mean.npy"
    std_file = info_path + "std.npy"
    model_file = info_path + "model.hdf5"

    # load mean + std + model
    mean = np.load(mean_file)
    std = np.load(std_file)
    model = load_model(model_file)

    # select subjects for this run
    subjects = all_subjects[run]

    # set class0
    class0 = "CN" if task == "AD" else "MCI-s"

    # get gradcam visualizations
    run_visualization(subjects, data_path, run_path, mean, std, model, class0, run)

    print('\nend')


def get_layer(model, vis_layer):
    """
    Returns the name of the conv3D layer to visualize
    which should be specified in the config file with an integer
    """
    conv_cnt = 0
    for layer in model.layers:
        if layer.name[:6] == 'conv3d':
            conv_cnt += 1
            if conv_cnt == vis_layer:
                layer_name = layer.name
    return layer_name


def run_visualization(subjects, data_path, run_path, mean, std, model, class0, run):
    """
    Performs a gradcam analysis of multiple subjects of a specified class.

        INPUT:
            subjects - the subject IDs of the subjects to be processed in this run
            data_path - the path in which the data of the subjects can be found
            run_path - the path in which all output of this run should be saved
            mean - the voxel-wise mean used for normalizing the images
            std - the voxel-wise std used for normalizing the images
            model - the model for which the gradcam should be computed
            class0 - which class is 0
            run - to which run this gradcam analysis belongs

        OUTPUT:
            saves the mean and variation of the Grad-CAM map averaged over the subjects in this run

    """

    # get image shape
    X, Y, Z = model.input_shape[1], model.input_shape[2], model.input_shape[3]

    # get name of layer to visualize
    gc_layer_name = get_layer(model, gc_layer)

    # write the subjects of this run in a txt file
    file = open(f"{run_path}subjects_run{run}.txt", 'w')
    file.write(f"\nRUN {run} - Analysis of {class_limit} {label} subjects\n")
    print(f"\nRUN {run} - Analysis of {class_limit} {classification_type} classified {label} subjects\n")

    # allocate batch of images
    images = np.zeros((len(subjects), X, Y, Z))

    # loop over all subjects in this run
    for i, subject in enumerate(subjects):

        # get image path
        img_file = f"{data_path}{subject}.npy"
        file.write(f"{i} - subject {subject}\n")
        print(f"\n{i} - Working on subject: {subject}")

        # load + normalize image and store in batch
        image = load_image(img_file, mean, std)
        images[i] = image

    # create array indicating the class of each images
    cls = 0 if label == class0 else 1
    classes = [cls] * len(subjects)

    # expand image batch to network input shape
    images = np.expand_dims(images, axis=5)

    # compute gradcam mean and variation of the subjects in this run
    gradcam_mean, gradcam_var = grad_cam_batch(model, images, classes, gc_layer_name, X, Y, Z)

    # if last run contains less subjects than class limit these should weigh less in the total average
    if len(subjects) is not class_limit:
        weight_factor = len(subjects) / class_limit
        gradcam_mean = gradcam_mean * weight_factor
        gradcam_var = gradcam_var * weight_factor

    # save mean + var gradcams
    np.save(f"{run_path}gradcam_c{gc_layer}_{label}.npy", gradcam_mean)
    np.save(f"{run_path}gradcam-VAR_c{gc_layer}_{label}.npy", gradcam_var)

    file.close()


def load_image(img_file, mean, std):
    """
    Load and normalize image with input mean and std.
    """

    x = np.load(img_file)
    x = np.subtract(x, mean)
    x = np.divide(x, (std + 1e-10))

    return x


def grad_cam_batch(model, images, classes, layer_name, X, Y, Z):
    """
    GradCAM method to process multiple images in one run.

        INPUT:
            model - the model for which the gradcams should be computed
            images - the batch of images for which the gradcams should be computed
            classes - an array indicating the classes corresponding to the image batch
            X, Y, Z - the input shapes of the images

        OUTPUT
            cam_mean - the average gradcam of the image batch
            cam_var - the variation of the mean gradcam of the image batch
    """

    # get loss, output and gradients
    loss = tf.gather_nd(model.output, np.dstack([range(images.shape[0]), classes])[0])
    layer_output = model.get_layer(layer_name).output
    grads = K.gradients(loss, layer_output)[0]

    # create gradient function
    gradient_fn = K.function([model.input, K.learning_phase()], [layer_output, grads])

    # calculate class activation maps for image batch
    conv_output, grads_val = gradient_fn([images, 0])
    weights = np.mean(grads_val, axis=(1, 2, 3))
    cams = np.einsum('ijklm,im->ijkl', conv_output, weights)

    # process CAMs
    new_cams = np.empty((images.shape[0], X, Y, Z))
    for i in range(new_cams.shape[0]):
        cam_i = cams[i] - cams[i].mean()
        cam_i = (cam_i + 1e-10) / (np.linalg.norm(cam_i) + 1e-10)
        new_cams[i] = resize(cam_i, (X, Y, Z), order=1, mode='constant', cval=0, anti_aliasing=False)
        new_cams[i] = np.maximum(new_cams[i], 0)
        new_cams[i] = new_cams[i] / new_cams[i].max()

    # calculate mean and variation
    cam_mean = np.mean(new_cams, axis=0)
    cam_var = np.var(new_cams, axis=0)

    return cam_mean, cam_var


def create_data_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    main()
