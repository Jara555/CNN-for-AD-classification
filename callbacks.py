import config
import math
import keras
from keras.callbacks import LearningRateScheduler
from keras.callbacks import Callback
from EpochPerformance import EpochPerformance


def step_decay(epoch):
    """
    Calculate new learning rate based on a drop value and a drop per epochs value.
    This way the learning rate will be decreased by a step wise decay.
    """

    lrate = config.lr * math.pow(config.lr_drop, math.floor((1 + epoch) / config.lr_epochs_drop))
    return lrate


def exp_decay(epoch):
    """
    Calculate new learning rate based on a constant value k.
    This way the learning rate will be decreased by exponential decay.
    """
    k = 0.1
    lrate = config.lr * math.exp(-k * epoch)
    return lrate


class LossHistory(Callback):
    """
    Saves the learning rate in history
    """
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(step_decay(len(self.losses)))


def callbacks_list(CM_train_generator, CM_validation_generator, labels, results_dir):
    """
    Creates a list with callbacks.
        - Model checkpoints (weight improvement)
        - Tensorboard
        - Early stopping
        - Learning rate scheduler
        - Performance per epoch
    """

    # tensorboard
    tb_callback = keras.callbacks.TensorBoard(log_dir=results_dir + '/logs', histogram_freq=0, batch_size=config.batch_size,
                                              write_graph=False, write_grads=True, write_images=False, embeddings_freq=0,
                                              embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None,
                                              update_freq='epoch')

    # early stopping based on acc
    es_callback = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.01,
                                                patience=config.es_patience, verbose=1, mode='auto', baseline=None)

    # learning rate scheduler
    lr_scheduler = LearningRateScheduler(step_decay)
    loss_history = LossHistory()

    # epoch performance: monitors sens, spec, AUC + acc after each epoch
    # saves best model based on AUC + early stopping based on AUC improvement
    train_epoch_performance = EpochPerformance(generator=CM_train_generator, labels=labels, name="trn", results_dir=results_dir)
    val_epoch_performance = EpochPerformance(generator=CM_validation_generator, labels=labels, name="val", results_dir=results_dir)

    # save best model based on acc
    filepath = results_dir + "/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    acc_checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                                     save_weights_only=True, mode='auto', period=1)

    # create and return list
    callbacks = []
    if config.epoch_performance:
        callbacks.append(train_epoch_performance)
        callbacks.append(val_epoch_performance)
    if config.tensorboard:
        callbacks.append(tb_callback)
    if config.acc_checkpoint:
        callbacks.append(acc_checkpoint)
    if config.acc_early_stopping:
        callbacks.append(es_callback)
    if config.lr_scheduler:
        callbacks.append(lr_scheduler)
        callbacks.append(loss_history)

    return callbacks

