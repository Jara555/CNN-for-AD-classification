# import configuration file
import config

# set random seed
from numpy.random import seed
from tensorflow import set_random_seed
seed(config.fixed_seed)
set_random_seed(config.fixed_seed)

import keras
from keras import Input, Model
from keras.layers import Conv3D, BatchNormalization, Activation, Dropout, GlobalAveragePooling3D


def build_model_allCNN():
    """
    Builds a 3D all-CNN model which can be used for AD classification based on MRI.

        OUTPUT:
            model - the Keras implementation of the all-CNN
    """

    # INPUT
    input_image = Input(shape=(config.input_shape[0], config.input_shape[1], config.input_shape[2], 1))

    # use smaller model for down sampled data
    if not config.WB:

        # BLOCK 1
        x = Conv3D(filters=8, kernel_size=(3, 3, 3), padding='same', kernel_regularizer=config.weight_regularize)(input_image)
        x = Dropout(config.dropout)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # pooling
        x = Conv3D(filters=8, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same', kernel_regularizer=config.weight_regularize)(x)
        x = Dropout(config.dropout)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # BLOCK 2
        x = Conv3D(filters=16, kernel_size=(3, 3, 3), padding='same', kernel_regularizer=config.weight_regularize)(x)
        x = Dropout(config.dropout)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # pooling
        x = Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same', kernel_regularizer=config.weight_regularize)(x)
        x = Dropout(config.dropout)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # BLOCK 3
        x = Conv3D(filters=24, kernel_size=(3, 3, 3), padding='same', kernel_regularizer=config.weight_regularize)(x)
        x = Dropout(config.dropout)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # pooling
        x = Conv3D(filters=24, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same', kernel_regularizer=config.weight_regularize)(x)
        x = Dropout(config.dropout)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # BLOCK 4
        x = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', kernel_regularizer=config.weight_regularize)(x)
        x = Dropout(config.dropout)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # pooling
        x = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same', kernel_regularizer=config.weight_regularize)(x)
        x = Dropout(config.dropout)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    # use more blocks and kernels for whole brain data
    else:

        # BLOCK 1
        x = Conv3D(filters=16, kernel_size=(3, 3, 3), padding='same', kernel_regularizer=config.weight_regularize)(input_image)
        x = Dropout(config.dropout)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # pooling
        x = Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same', kernel_regularizer=config.weight_regularize)(x)
        x = Dropout(config.dropout)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # BLOCK 2
        x = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', kernel_regularizer=config.weight_regularize)(x)
        x = Dropout(config.dropout)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # pooling
        x = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same', kernel_regularizer=config.weight_regularize)(x)
        x = Dropout(config.dropout)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # BLOCK 3
        x = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', kernel_regularizer=config.weight_regularize)(x)
        x = Dropout(config.dropout)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # pooling
        x = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same', kernel_regularizer=config.weight_regularize)(x)
        x = Dropout(config.dropout)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # BLOCK 4
        x = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', kernel_regularizer=config.weight_regularize)(x)
        x = Dropout(config.dropout)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # pooling
        x = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same', kernel_regularizer=config.weight_regularize)(x)
        x = Dropout(config.dropout)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # BLOCK 5
        x = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', kernel_regularizer=config.weight_regularize)(x)
        x = Dropout(config.dropout)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # pooling
        x = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same', kernel_regularizer=config.weight_regularize)(x)
        x = Dropout(config.dropout)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # BLOCK 6
        x = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', kernel_regularizer=config.weight_regularize)(x)
        x = Dropout(config.dropout)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # pooling
        x = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same', kernel_regularizer=config.weight_regularize)(x)
        x = Dropout(config.dropout)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    # BLOCK 7: 1-by-1 convolutions
    x = Conv3D(filters=16, kernel_size=(1, 1, 1), padding='same', kernel_regularizer=config.weight_regularize)(x)
    x = Dropout(config.dropout)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # pooling
    x = Conv3D(filters=16, kernel_size=(1, 1, 1), strides=(2, 2, 2), padding='same', kernel_regularizer=config.weight_regularize)(x)
    x = Dropout(config.dropout)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # LAST: 1-by-1 convoluation + kernel size of 2
    x = Conv3D(filters=2, kernel_size=(1, 1, 1), padding='valid')(x)
    x = Dropout(config.dropout)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # gets input of shape (2,2,2,2) and converts this to output with shape (2,)
    x = GlobalAveragePooling3D()(x)

    # OUTPUT
    predictions = Activation('softmax')(x)

    model = Model(inputs=input_image, outputs=predictions)
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=config.lr, beta_1=0.9, beta_2=0.999, epsilon=config.epsilon, decay=config.decay, amsgrad=False),
                  metrics=['accuracy'])

    return model

