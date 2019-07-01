# import configuration file
import config

# set random seed
from numpy.random import seed
from tensorflow import set_random_seed
seed(config.fixed_seed)
set_random_seed(config.fixed_seed)

import csv
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold


def create_labels():
    """
    Reading & formatting of the subject IDs and labels

        INPUT:
            subjectfile - should be a .csv file with subject IDs in col 0 and labels in col 1

        OUTPUT:
            partition_labels - format {"CN" : [001, 002, 003], "AD" : [004, 005 ...] }
            labels - format {"001" : 0 , "002" : 1, "003" : 0 ...}
    """

    # create dicts
    partition_labels = {config.class0: [], config.class1: []}
    labels = dict()

    # open csv file and write 0 for class0 (CN / MCI-s) and 1 for class1 (AD / MCI-c)
    file = open(config.subjectfile, 'r')
    for line in csv.reader(file):
        if line[1] == config.class0:
            partition_labels[line[1]].append(line[0])
            labels[line[0]] = 0
        elif line[1] == config.class1:
            partition_labels[line[1]].append(line[0])
            labels[line[0]] = 1

    file.close()

    print("\nCATEGORIES\n")
    for cat in partition_labels:
        print("    " + cat + " : " + str(len(partition_labels[cat])))

    return partition_labels, labels


def split_train_test(partition_labels, labels):
    """
    Splits a dataset in a training and test set, based on stratified K fold

        INPUT:
            partition_labels, labels - see 'create_labels()'

        OUTPUT:
            partition_train_test[k] - subjects split in train and test group
            format {"train" : [001, 002, 003], "test" : [004, 005, ...] }
    """

    partition_train_test = {"train": [], "test": []}

    # get X (subjects) and corresponding y (labels)
    X = np.concatenate((partition_labels[config.class0], partition_labels[config.class1]), axis=0)
    y = np.array([0] * len(partition_labels[config.class0]) + [1] * len(partition_labels[config.class1]))

    # create k training and test sets (for stratified k cross validations)
    if config.shuffle_split:
        # random distribution of subjects over train and test sets
        skf = StratifiedShuffleSplit(n_splits=config.k_cross_validation, test_size=config.test_size, random_state=config.train_val_test_seed)
    else:
        # k folds: every subject in test set once
        skf = StratifiedKFold(n_splits=config.k_cross_validation, shuffle=True, random_state=config.train_val_test_seed)

    # split based on X and y
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        partition_train_test["train"].append(X_train)
        partition_train_test["test"].append(X_test)

    print("\nTRAIN TEST")
    count_sets(partition_train_test, labels)

    return partition_train_test


def split_train_val(partition_train_test, labels):
    """
    Splits a dataset in a training and validation set, based on stratified K folds

        INPUT:
            partition_train_test - train val should be extracted from training set only
            labels - see 'create_labels()'

        OUTPUT:
            partition_train_validation[k] - subjects split in train and val group
            format {"train" : [001, 002, 003], "validation" : [004, 005 ...] }
    """

    partition_train_validation = {"train": [], "validation": []}

    # for k-fold times
    for i in range(config.k_cross_validation):

        # regroup training set based on labels
        temp = {0: [], 1: []}
        for id in partition_train_test["train"][i]:
            temp[labels[id]].append(id)

        # create X (subjects) and y (labels)
        X = np.concatenate((temp[0], temp[1]), axis=0)
        y = np.array([0] * len(temp[0]) + [1] * len(temp[1]))

        # create stratified training and test set for that fold
        skf = StratifiedShuffleSplit(n_splits=1, test_size=config.val_size, random_state=config.train_val_test_seed)
        for train_index, validation_index in skf.split(X, y):
            X_train, X_validation = X[train_index], X[validation_index]
            partition_train_validation["train"].append(X_train)
            partition_train_validation["validation"].append(X_validation)

    print("\nTRAIN VALIDATION")
    count_sets(partition_train_validation, labels)

    return partition_train_validation


def count_sets(dic, labels):
    """
    Counts and prints the amount of sets present per k-fold and class

        INPUT:
            dic - dictionary of type dic["set"]["k-fold"]["id"]
            labels - dictionary of type labels["id"]["class"]

        OUTPUT:
            print overview of set distributions
    """

    # loop over set types (train/test)
    for set in dic:
        print("\n" + set)

        # loop over k-folds
        for i in range(len(dic[set])):
            a = []

            # loop over ids
            for id in dic[set][i]:
                a.append(labels[id])
            unique, counts = np.unique(a, return_counts=True)

            # replace with real labels
            c = []
            for u in unique:
                if u == 0:
                    c.append(config.class0)
                elif u == 1:
                    c.append(config.class1)

            # print distribution
            r = dict(zip(c, counts))
            print("    fold " + str(i) + ":", r)
