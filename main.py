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
from model_selection import select_model, load_best_model
from sklearn.metrics import roc_curve, auc, classification_report
from savings import save_results, save_DL_model
from create_sets import create_labels, split_train_test, split_train_val, count_sets
from generator import DataGenerator
from standardize import standardization_matrix
from plotting import plot_acc_loss, plot_ROC, plot_epoch_performance
from augmentation import augmentation
from callbacks import callbacks_list


def main(argv):
    """
    This script will implement the training and testing of a deep learning model for the classification of
    Alzheimer's disease, based on structural MRI data. All settings and model parameters can be defined in the
    config.py configuration file.

    In short, data is split in a test, train and validation set. Cross-validation is performed for which in
    each fold data is augmented, normalized and a CNN is trained and evaluated. As output this script provides train,
    validation & test performance metrics, which are all stored in a dictionary called 'results.npy'. Plots of the
    performance per epoch and of the ROC-AUC of all folds will be provided. Also the configurations file will be saved,
    together with the model information.

    Live performance can be monitored through TensorBoard (only when running local)
    $ tensorboard --logdir=<results dir>/logs
    """

    # start timer
    start = time.time()
    start_localtime = time.localtime()

    # if temp job dir is provided as input use this as data dir (server)
    if len(argv) > 1:
        config.data_dir = sys.argv[1] + "/"
        config.aug_dir = sys.argv[1] + "/augmented/"
    # if job nr is provided use this to define output dir (server)
    if len(argv) > 2:
        config.output_dir = f"{config.all_results_dir}{sys.argv[2]}_{config.roi}_{config.task}_{config.model}{config.comments}/"

    # save configuration file
    create_data_directory(config.output_dir)
    copyfile(config.config_file, f"{config.output_dir}configuration_{config.model}.py")

    # initialization of results dictionary
    results = {"train": {"loss": [], "acc": [], "fpr": [], "tpr": [], "auc": [], "sensitivity": [], "specificity": []},
               "validation": {"loss": [], "acc": [], "fpr": [], "tpr": [], "auc": [], "sensitivity": [], "specificity": []},
               "test": {"loss": [], "acc": [], "fpr": [], "tpr": [], "auc": [], "sensitivity": [], "specificity": []}}

    # create labels
    partition_labels, labels = create_labels()

    # train test split
    partition_train_test = split_train_test(partition_labels, labels)
    np.save(config.output_dir + "train_test.npy", partition_train_test)

    # train val split
    partition_train_validation = split_train_val(partition_train_test, labels)
    np.save(config.output_dir + "train_val.npy", partition_train_validation)

    # START CROSS VALIDATION
    for i in range(config.k_cross_validation):

        # select model
        model = select_model(i)

        print("\n----------- CROSS VALIDATION " + str(i) + " ----------------\n")

        # augmentation of training data
        if config.augmentation:
            partition_train_validation["train"][i], labels = augmentation(partition_train_validation["train"][i], labels)
            count_sets(partition_train_validation, labels)

        # create results directory for fold
        results_dir = config.output_dir + "k" + str(i)
        create_data_directory(results_dir)
        file = open(results_dir + "/results.txt", 'w')

        # get mean + std of train data to standardize all data in generator
        if config.all_data:
            mean = np.load(config.mean_file)
            std = np.load(config.std_file)
        else:
            mean, std = standardization_matrix(partition_train_validation["train"][i])

        # save mean + std
        np.save(results_dir + "/mean.npy", mean)
        np.save(results_dir + "/std.npy", std)

        # create data generators
        train_generator = DataGenerator(partition_train_validation["train"][i], labels, mean, std,
                                        batch_size=config.batch_size, dim=config.input_shape, n_channels=1, n_classes=2, shuffle=True)
        validation_generator = DataGenerator(partition_train_validation["validation"][i], labels, mean, std,
                                             batch_size=config.batch_size, dim=config.input_shape, n_channels=1, n_classes=2, shuffle=True)
        test_generator = DataGenerator(partition_train_test["test"][i], labels, mean, std,
                                       batch_size=1, dim=config.input_shape, n_channels=1, n_classes=2, shuffle=False)
        CM_train_generator = DataGenerator(partition_train_validation["train"][i], labels, mean, std,
                                           batch_size=1, dim=config.input_shape, n_channels=1, n_classes=2, shuffle=False)
        CM_validation_generator = DataGenerator(partition_train_validation["validation"][i], labels, mean, std,
                                                batch_size=1, dim=config.input_shape, n_channels=1, n_classes=2, shuffle=False)

        # set callbacks
        callback_list = callbacks_list(CM_train_generator, CM_validation_generator, labels, results_dir)

        if not config.test_only:

            # TRAINING
            history = model.fit_generator(generator=train_generator, validation_data=validation_generator,
                                          class_weight=None, callbacks=callback_list, epochs=config.epochs, verbose=1,
                                          use_multiprocessing=False, workers=0)

            # plot acc + loss
            plot_acc_loss(history, results_dir, i)

            # plot performance per epoch
            if config.epoch_performance:
                plot_epoch_performance(callback_list[0])
                plot_epoch_performance(callback_list[1])

            # load model of epoch with best performance
            model = load_best_model(results_dir)

            # TRAIN EVALUATION

            # roc auc
            Y_pred = model.predict_generator(CM_train_generator, verbose=0)
            y_pred = np.argmax(Y_pred, axis=1)
            y_true = []
            for id in CM_train_generator.list_IDs:
                y_true.append(labels[id])
            fpr, tpr, thresholds = roc_curve(y_true, Y_pred[:,1])
            roc_auc = auc(fpr, tpr)

            # save classification per subject (for statistical test)
            np.save(results_dir + "/train_IDs.npy", CM_train_generator.list_IDs)
            np.save(results_dir + "/train_y_true.npy", y_true)
            np.save(results_dir + "/train_y_pred.npy", y_pred)

            # sen / spe
            report = classification_report(y_true, y_pred, target_names=[config.class0, config.class1], output_dict=True)

            # loss, acc
            score = model.evaluate_generator(generator=train_generator, verbose=1)

            results["train"]["loss"].append(score[0])
            results["train"]["acc"].append(score[1])
            results["train"]["fpr"].append(fpr)
            results["train"]["tpr"].append(tpr)
            results["train"]["auc"].append(roc_auc)
            results["train"]["sensitivity"].append(report[config.class1]["recall"])
            results["train"]["specificity"].append(report[config.class0]["recall"])

            # report train results
            train_results = f"\nTrain\n    loss: {score[0]:.4f}\n    acc: {score[1]:.4f}\n    AUC: {roc_auc:.4f}\n    " \
                            f"sens: {report[config.class1]['recall']:.4f}\n    spec: {report[config.class0]['recall']:.4f}\n\n"
            file.write(train_results), print(train_results)

            # VALIDATION EVALUATION

            # roc auc
            Y_pred = model.predict_generator(CM_validation_generator, verbose=0)
            y_pred = np.argmax(Y_pred, axis=1)
            y_true = []
            for id in CM_validation_generator.list_IDs:
                y_true.append(labels[id])
            fpr, tpr, thresholds = roc_curve(y_true, Y_pred[:,1])
            roc_auc = auc(fpr, tpr)

            # save classification per subject (for statistical test)
            np.save(results_dir + "/val_IDs.npy", CM_validation_generator.list_IDs)
            np.save(results_dir + "/val_y_true.npy", y_true)
            np.save(results_dir + "/val_y_pred.npy", y_pred)

            # sen / spe
            report = classification_report(y_true, y_pred, target_names=[config.class0, config.class1], output_dict=True)

            # loss, acc
            score = model.evaluate_generator(generator=validation_generator, verbose=1)

            results["validation"]["loss"].append(score[0])
            results["validation"]["acc"].append(score[1])
            results["validation"]["fpr"].append(fpr)
            results["validation"]["tpr"].append(tpr)
            results["validation"]["auc"].append(roc_auc)
            results["validation"]["sensitivity"].append(report[config.class1]["recall"])
            results["validation"]["specificity"].append(report[config.class0]["recall"])

            # report val results
            val_results = f"\nValidation\n    loss: {score[0]:.4f}\n    acc: {score[1]:.4f}\n    AUC: {roc_auc:.4f}\n    " \
                            f"sens: {report[config.class1]['recall']:.4f}\n    spec: {report[config.class0]['recall']:.4f}\n\n"
            file.write(val_results), print(val_results)

        # TEST EVALUATION

        # roc auc
        Y_pred = model.predict_generator(test_generator, verbose=0)
        y_pred = np.argmax(Y_pred, axis=1)
        y_true = []
        for id in test_generator.list_IDs:
            y_true.append(labels[id])
        fpr, tpr, thresholds = roc_curve(y_true, Y_pred[:,1])
        roc_auc = auc(fpr, tpr)

        # save classification per subject (for statistical test)
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

        # delete augmented images
        if config.augmentation:
            os.system('rm -rf %s/*' % config.aug_dir)

    print("\n---------------------- RESULTS ----------------------\n\n")

    # plot test ROC of all folds + average + std
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
