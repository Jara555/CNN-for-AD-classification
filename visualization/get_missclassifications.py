import csv
import os
import sys
from random import shuffle
import math
import numpy as np
from keras.engine.saving import load_model

# import configuration parameters
from visualization.vis_config import roi, task, label, data_set, val, server, temp, class_limit


def main():
    """
    This script should be run before 'vis_main.py' to get a list of correctly and miss classified subjects for
    a specific class. These lists are saved in the info path for compiling the Grad-CAM images and are used
    for the class-averaged computation of these images.

    Because not more than 10 images can be compiled at once the lists with subjects are build up as multiple
    seperate lists: e.g. corr_subs[i] in which i is the amount of seperate subject splits. In this way the
    'vis_main.py' script can be run several times using every split with subjects and can finally average the
    Grad-CAMs of these splits with 'vis_average.py'.

    A Grad-CAM info directory should be created in which the model, mean, std and train-test files are stored.
    These will be accessed by the current visualization scripts. This script will store several subject files
    in this info folder including all the correct and miss classified subjects per class, which can be accessed
    by the other visualization scripts.

    """

    # set paths (when running on server based on data dir + job nr input)
    if server:
        data_path = sys.argv[1] + "/" if temp else f"/path/to/data/data_{roi}/"
        info_path = f"/local/path/to/gradcam/info/{roi}/{task}/info/"
        subjectfile = "/local/path/to/labels/AllSubjectsDiagnosis.csv"
    else:
        data_path = f"/server/path/to/data/data_{roi}/"
        info_path = f"/server/path/to/gradcam/info/{roi}/{task}/info/"
        subjectfile = "/server/path/to/labels/AllSubjectsDiagnosis.csv"

    # set model information files
    mean_file = info_path + "mean.npy"
    std_file = info_path + "std.npy"
    model_file = info_path + "model.hdf5"
    set_file = f"{info_path}train_val.npy" if val or data_set == "train" else f"{info_path}train_test.npy"

    # set classes
    if task == "AD":
        class0 = "CN"
        class1 = "AD"
    else:
        class0 = "MCI-s"
        class1 = "MCI-c"

    # split subjects based on classes: {"AD": [1, 2, ...], "CN" : [3, 4, ...]}
    classes_split = split_classes(subjectfile, set_file, class0, class1)

    # get mean + std + model
    mean = np.load(mean_file)
    std = np.load(std_file)
    model = load_model(model_file)

    # get model predictions
    corr_class, miss_class = count_missclass(classes_split[label], data_path, mean, model, std, class0, class1)

    # split subjects based on correct or miss classification
    corr_subs = split_subjects(corr_class)
    miss_subs = split_subjects(miss_class)

    print(f"\nCORRECT: {len(corr_subs)} splits of {len(corr_subs[0])} subjects")
    print(f"\nMISS: {len(miss_subs)} splits of {len(miss_subs[0])} subjects")

    # save the correct and miss classified subjects of a model for a specific class
    np.save(f"{info_path}correct_classified_subjects_{label}_{data_set}.npy", corr_subs)
    np.save(f"{info_path}miss_classified_subjects_{label}_{data_set}.npy", miss_subs)

    print('\nend')


def split_classes(subjectfile, set_file, class0, class1):
    """
    Splits subjects of the test group in 2 classes
    Returns a dictionary: {"AD": [sub1, sub2, ...], "CN": [sub5, sub6, ...]}
    """
    # get test subjects
    subs = np.load(set_file, allow_pickle=True).item()[data_set][1]
    print(f"\nIn total {len(subs)} subjects present in {data_set} set")

    # split subjects based on class
    classes_split = {class0: [], class1: []}
    file = open(subjectfile, 'r')
    for line in csv.reader(file):
        if line[1] == class0 and line[0] in subs:
            classes_split[line[1]].append(line[0])
        elif line[1] == class1 and line[0] in subs:
            classes_split[line[1]].append(line[0])

    print("\nCATEGORIES")
    for cat in classes_split:
        print(f"    {cat} : {len(classes_split[cat])}")

    shuffle(classes_split[class0])
    shuffle(classes_split[class1])

    file.close()

    return classes_split


def split_subjects(subjects):
    """
    Splits the list of subjects in k splits.
    Needed when running on the server with WB data, which memory allows max 10 subs at once.
    By splitting the subjects the script can run multiple times, at the end the average is calculated.
    """
    subs_split = []

    if class_limit > len(subjects):
        k_splits = 1
    else:
        k_splits = math.ceil(len(subjects) / class_limit)

    # split subjects in k lists of 10
    for k in range(1, k_splits + 1):
        subs_split.append(subjects[k*class_limit-class_limit:k*class_limit])

    return subs_split


def count_missclass(subjects, data_path, mean, model, std, class0, class1):
    """
    Get the model predictions for multiple input images. Splits the correctly and miss classified
    subjects based on these predictions.
    """

    corr_class = []
    miss_class = []

    for i, subject in enumerate(subjects):

        img_file = f"{data_path}{subject}.npy"
        print(f"\n{i} - Working on subject: {subject} - with true label: {label}")

        # load + standardize image
        image = load_image(img_file, mean, std)

        exp_im = np.expand_dims(image, axis=0)
        exp_im = np.expand_dims(exp_im, axis=5)
        predictions = model.predict(exp_im)
        cls = np.argmax(predictions)
        class_name = class0 if cls == 0 else class1
        print(f'\tModel prediction:\n\t\t{class_name}\twith probability {predictions[0][cls]:.4f}')

        # only process if correct classification
        if class_name == label:
            corr_class.append(subject)
        else:
            print("Miss classification")
            miss_class.append(subject)

    update = f"\n{label} - correct classifications: {len(corr_class)} - miss classifications: {len(miss_class)}\n\n"
    print(update)

    return corr_class, miss_class


def load_image(img_file, mean, std):
    """Load and normalize image"""

    x = np.load(img_file)
    x = np.subtract(x, mean)
    x = np.divide(x, (std + 1e-10))

    return x


def create_data_directory(path):
    """
    Create data path when not existed yet.
    """
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    main()
