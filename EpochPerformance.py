import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, roc_auc_score

import config


class EpochPerformance(Callback):
    """
    This callback class keeps track of the sensitivity, specificity, accuracy and AUC after each epoch.
    This can be used for both the training and validation data.

    The model with the best AUC score is saved, for later evaluation. When no AUC improvement after a
    specified amount of epochs early stopping is induced.
    """

    def __init__(self, generator, labels, name, results_dir):
        super(Callback).__init__()
        self.generator = generator
        self.labels = labels
        self.name = name
        self.results_dir = results_dir

    def on_train_begin(self, logs={}):
        self.auc_list = []
        self.sens_list = []
        self.spec_list = []
        self.acc_list = []
        self.best_epoch = 0

        return

    def on_epoch_end(self, epoch, logs={}):
        """
        After every epoch during training calculate the AUC, sens, spec and acc.
        Save model when AUC is improved and induce early stopping when necessary.
        """

        # get true labels and predicted labels
        Y_pred = self.model.predict_generator(self.generator, verbose=0)
        y_pred = np.argmax(Y_pred, axis=1)
        y_true = []
        for id in self.generator.list_IDs:
            y_true.append(self.labels[id])

        # calculate auc based on prediction estimates
        auc = roc_auc_score(y_true, Y_pred[:,1])

        # calculate acc, sens + spec based on confusion matrix
        conf = confusion_matrix(y_true, y_pred)
        acc = np.trace(conf) / float(np.sum(conf))
        sens = conf[1, 1] / (conf[1, 1] + conf[1, 0])
        spec = conf[0, 0] / (conf[0, 0] + conf[0, 1])

        print(f"Epoch {epoch+1:03d}: {self.name}_auc: {auc:.4f} - {self.name}_sens: {sens:.4f} - {self.name}_spec: {spec:.4f} - {self.name}_acc: {acc:.4f}")

        self.auc_list.append(auc)
        self.sens_list.append(sens)
        self.spec_list.append(spec)
        self.acc_list.append(acc)

        # keep track of validation AUC for best model + early stopping
        if self.name is "val":

            # save model if best validation auc
            if auc is max(self.auc_list):
                filepath = self.results_dir + f"/weights-improvement-e{epoch+1:02d}-auc{auc:.2f}.hdf5"
                self.model.save(filepath)
                self.best_epoch = epoch
                print(f"Epoch {epoch+1:03d}: {self.name}_auc improved to {auc:.4f}, saving model to {filepath}\n")
            else:
                no_improvement = epoch - self.best_epoch
                print(f"Epoch {epoch+1:03d}: {self.name}_auc did not improve from {max(self.auc_list):.4f} for {no_improvement:03d} epochs\n")

                # stop training after # epochs without improvement
                if config.early_stopping and no_improvement is config.es_patience:
                    print(f"Epoch {epoch+1:03d}: {no_improvement:03d} epochs without AUC improvement - early stopping\n")
                    self.model.stop_training = True

        return

    def on_train_end(self, logs={}):
        return

