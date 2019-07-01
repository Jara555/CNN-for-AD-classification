# import configuration file
import config

# set random seed
from numpy.random import seed
from tensorflow import set_random_seed
seed(config.fixed_seed)
set_random_seed(config.fixed_seed)

import os
import sys
import keras

from keras import optimizers
from keras.engine.saving import load_model
from model_allCNN import build_model_allCNN


def select_model(i):
    """
    Selects and returns the model to use based on the fold and the information of the config file.

        INPUT:
            i - the fold of the cross validation

        OUTPUT:
            model - the selected model

    """

    # load pre-train model
    if config.pre_train:
        model = load_model(config.pre_train_model)
    # build new CNN
    else:
        if config.model == "allCNN":
            model = build_model_allCNN()
        else:
            sys.exit("No valid model selected")

    # freeze first part of network
    if config.freeze:
        model.trainable = True
        set_trainable = False
        conv_cnt = 0
        # select layer to freeze until
        for layer in model.layers:
            if layer.name[:6] == 'conv3d':
                conv_cnt += 1
                # make model trainable from this layer on
                if conv_cnt == config.freeze_until:
                    set_trainable = True
                    layer_name = layer.name
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False

        model.compile(loss=keras.losses.binary_crossentropy, optimizer=optimizers.RMSprop(lr=1e-5),
                      metrics=['accuracy'])

    # print model summary for first fold
    if i == 0:
        model.summary()

    if config.freeze:
        print(f"\nModel is frozen until layer {config.freeze_until} (with layer name: {layer_name})\n")
    if config.pre_train and config.test_only:
        print("\nNo training -> testing only!\n")

    return model


def load_best_model(results_dir):
    """
    Select the best model of the list of model checkpoints saved in the results dir of a fold.

        INPUT:
            results_dir - the directory in which the saved model checkpoints can be found

        OUTPUT:
            model - the best performing trained Keras model which was saved as checkpoint
    """

    # get all files in the results dir and sort so that best model is listed last
    L = os.listdir(results_dir)
    L.sort()

    # select best model
    model = load_model(results_dir + "/" + L[-1])
    print("Loading best model for evaluation: ", L[-1])

    return model

