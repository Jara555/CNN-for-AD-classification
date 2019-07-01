# import configuration file
import config

# set random seed
from numpy.random import seed
from tensorflow import set_random_seed
seed(config.fixed_seed)
set_random_seed(config.fixed_seed)

import tensorflow as tf
import os
from keras import backend as K

# when running at local pc avoid using all GPU memory for 1 experiment
if config.location == "local":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    sc = tf.ConfigProto()
    sc.gpu_options.allow_growth = True
    s = tf.Session(config = sc)
    K.set_session(s)

import os
import time
import numpy as np
import sys
from shutil import copyfile
from model_selection import load_best_model
from sklearn.metrics import roc_curve, auc, classification_report
from savings import save_results, save_DL_model
from create_sets import create_labels, count_sets
from generator import DataGenerator
from plotting import plot_ROC


def main(argv):
    """
    This script is an alternative version of 'main.py' which has the purpose of running a complete data set
    through a pre-trained model as evaluation only. This can be used for MCI classification, since for this
    task it is beneficial to use a network which is pre-trained on the AD task. For every model created in
    each fold of the cross-validation of the AD task, all MCI data is used to evaluate this model. In this
    way the MCI results of each fold can be averaged for the final performance.

    Similar as for the 'main.py' script this script uses the configuration settings of the 'config.py' file.
    Also this script saves all evaluation results in a 'results.npy' dictionary and provides a plot of the
    ROC-AUC of all folds. The configuration file and model information will be saved.
    """

    # start timer
    start = time.time()
    start_localtime = time.localtime()

    # if temp job directory is provided, use this as data direction (when running on server)
    if len(argv) > 1:
        config.data_dir = sys.argv[1] + "/"
        config.aug_dir = sys.argv[1] + "/augmented/"
    # if job nr is provided, use as output dir name
    if len(argv) > 2:
        config.output_dir = config.all_results_dir + sys.argv[2] + "_" + config.roi \
                            + "_" + config.task + "_" + config.model + config.comments + "/"

    # save configuration file
    create_data_directory(config.output_dir)
    copyfile(config.config_file, config.output_dir + "configuration_" + config.model + ".py")

    # initialization
    results = {"train": {"loss": [], "acc": [], "fpr": [], "tpr": [], "auc": [], "sensitivity": [], "specificity": []},
               "validation": {"loss": [], "acc": [], "fpr": [], "tpr": [], "auc": [], "sensitivity": [], "specificity": []},
               "test": {"loss": [], "acc": [], "fpr": [], "tpr": [], "auc": [], "sensitivity": [], "specificity": []}}

    # create labels and test set with all data
    partition_labels, labels = create_labels()
    partition_test = {"test": []}
    X = np.concatenate((partition_labels[config.class0], partition_labels[config.class1]), axis=0)
    for i in range(config.k_cross_validation):
        partition_test["test"].append(X)
    count_sets(partition_test, labels)
    print("\n")
    np.save(config.output_dir + "train_test.npy", partition_test)

    # START CROSS VALIDATION

    for i in range(config.k_cross_validation):

        # select pre-trained model of the specified fold
        fold_path = f"{config.pretrain_path}k{i}/"
        print(f"Location pre-trained model fold {i}: {fold_path}")
        model = load_best_model(fold_path)
        if i == 0:
            model.summary()
        if config.pre_train and config.test_only:
            print("\nNo training -> testing only!\n")

        print("\n----------- CROSS VALIDATION " + str(i) + " ----------------\n")

        # create results directory
        results_dir = config.output_dir + "k" + str(i)
        create_data_directory(results_dir)
        file = open(results_dir + '/results.txt', 'w')

        # get specified mean + std to standardize all data in generator
        mean = np.load(f"{fold_path}mean.npy")
        std = np.load(f"{fold_path}std.npy")

        # create data generator
        test_generator = DataGenerator(partition_test["test"][i], labels, mean, std, batch_size=1, dim=config.input_shape, n_channels=1, n_classes=2, shuffle=False)

        # TEST EVALUATION

        # roc auc
        Y_pred = model.predict_generator(test_generator, verbose=0)
        y_pred = np.argmax(Y_pred, axis=1)
        y_true = []
        for id in test_generator.list_IDs:
            y_true.append(labels[id])
        fpr, tpr, thresholds = roc_curve(y_true, Y_pred[:,1])
        roc_auc = auc(fpr, tpr)

        # save subject classifications (for statistical analysis)
        np.save(results_dir + "/test_IDs.npy", test_generator.list_IDs)
        np.save(results_dir + "/test_y_true.npy", y_true)
        np.save(results_dir + "/test_y_pred.npy", y_pred)

        # sen / spe
        report = classification_report(y_true, y_pred, target_names=[config.class0, config.class1], output_dict=True)

        # loss, acc
        score = model.evaluate_generator(generator=test_generator, verbose=1)

        results["test"]["loss"].append(score[0])
        results["test"]["acc"].append(score[1])
        results["test"]["fpr"].append(fpr)
        results["test"]["tpr"].append(tpr)
        results["test"]["auc"].append(roc_auc)
        results["test"]["sensitivity"].append(report[config.class1]["recall"])
        results["test"]["specificity"].append(report[config.class0]["recall"])

        # report test results
        test_results = f"\nTest\n    loss: {score[0]:.4f}\n    acc: {score[1]:.4f}\n    AUC: {roc_auc:.4f}\n    " \
                        f"sens: {report[config.class1]['recall']:.4f}\n    spec: {report[config.class0]['recall']:.4f}\n\n"
        file.write(test_results), print(test_results)
        file.close()

    print("\n---------------------- RESULTS ----------------------\n\n")

    # plot test ROC of all folds + average
    plot_ROC(results["test"]["tpr"], results["test"]["fpr"], results["test"]["auc"])

    # end timer
    end = time.time()
    end_localtime = time.localtime()

    # save results + model
    np.save(config.output_dir + "results.npy", results)
    save_DL_model(model)
    save_results(results, start, start_localtime, end, end_localtime)

    print('\nend')


def create_data_directory(path):
    """
    Creates new data path if not already exists
    """
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    main(sys.argv)
