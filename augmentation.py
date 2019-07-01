# import configuration file
import config

# set random seed
from numpy.random import seed
from tensorflow import set_random_seed
seed(config.fixed_seed)
set_random_seed(config.fixed_seed)

import os
import math
import numpy as np


def augmentation(dataset, labels):
    """
    Applies augmentation on a given dataset and appends the new images to the original dataset and labels

        INPUT:
            dataset, labels - original dataset and labels on which augmentation should be performed

        OUTPUT:
            dataset, labels - new sets including augmented images
            saves the augmented images in the given directory
    """

    print("Augmentation")

    # if necessary create aug dir and make sure it's empty
    if not os.path.exists(config.aug_dir):
        os.makedirs(config.aug_dir)
    else:
        os.system('rm -rf %s/*' % config.aug_dir)

    # sort ids based on category
    split_categories = {0: [], 1: []}
    for id in dataset:
        split_categories[labels[id]].append(id)

    # calculate the amount of missing images to be augmented
    missing = {0: max(0, config.class_total - len(split_categories[0])), 1: max(0, config.class_total - len(split_categories[1]))}
    print("    missing " + config.class0 + " data: ", missing[0])
    print("    missing " + config.class1 + " data: ", missing[1])

    cnt = 0

    # loop over categories
    for cat in split_categories:

        # loop over missing repetitions of whole dataset
        for rep_idx in range(math.floor(missing[cat] / len(split_categories[cat]))):

            # loop over ids in dataset
            for id in split_categories[cat]:

                aug_name = "aug" + str(cnt) + "_" + id

                # update labels + dataset
                labels[aug_name] = cat
                dataset = np.append(dataset, aug_name)

                # augment image + save
                aug_image = mixing(id, split_categories[cat])
                np.save(config.aug_dir + aug_name + ".npy", aug_image)

                cnt += 1

        # loop over rest of the missing images
        for rest_idx in range(missing[cat] % len(split_categories[cat])):

            id = split_categories[cat][rest_idx]
            aug_name = "aug" + str(cnt) + "_" + id

            # update labels + dataset
            labels[aug_name] = cat
            dataset = np.append(dataset, aug_name)

            # augment image + save
            aug_image = mixing(id, split_categories[cat])
            np.save(config.aug_dir + aug_name + ".npy", aug_image)

            cnt += 1

    return dataset, labels


def mixing(id, list_ids):
    """
    Applies augmentation on an image by mixing the original image with a random image of the same category.
    The augmentation factor indicates the percentage / fraction of the random image which will be used.

        INPUT:
            id - id of subject to be augmented
            list_ids - list with all ids to pick mix image

        OUTPUT:
            aug_image - the augmented image
    """

    # load original image
    image = np.load(config.data_dir + id + ".npy")

    # load random image from same category
    id_mix = np.random.choice(np.setdiff1d(list_ids, [id]))
    image_mix = np.load(config.data_dir + id_mix + ".npy")

    # mix images
    aug_image = (1 - config.aug_factor) * image + config.aug_factor * image_mix

    return aug_image




































