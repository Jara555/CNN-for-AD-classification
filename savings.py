# import configuration file
import config

import time
import numpy as np

from contextlib import redirect_stdout
from keras.utils import plot_model


def save_results(results, start, start_localtime, end, end_localtime):
    """
    This function manages the creation of a report in text format of the results of the cross-validation of
    a model. The average loss, acc, AUC, sens and spec are averaged over folds and reported for the evaluation
    of train, validation and test data. This final report is saved as 'results_cross_val.txt' in the output dir.
    """

    # save final results
    final_results = "RESULTS " + str(config.k_cross_validation) + "-FOLD CROSS VALIDATION\n\n" \
                    "Task: 		" + config.task + "\n" \
                    "Model: 		" + config.model + "\n" \
                    "ROI: 		" + config.roi + "\n" \
                    "Location: 	" + config.location + "\n" \
                    "Comments: 	" + config.comments + "\n\n" \
                    "TRAIN\n" \
                    "    loss: 	" + str(np.mean(results["train"]["loss"])) + "\n" \
                    "    acc: 	" + str(np.mean(results["train"]["acc"])) + "\n" \
                    "    AUC: 	" + str(np.mean(results["train"]["auc"])) + "\n" \
                    "    sens: 	" + str(np.mean(results["train"]["sensitivity"])) + "\n" \
                    "    spec:	" + str(np.mean(results["train"]["specificity"])) + "\n\n"\
                    "VALIDATION\n" \
                    "    loss: 	" + str(np.mean(results["validation"]["loss"])) + "\n" \
                    "    acc: 	" + str(np.mean(results["validation"]["acc"])) + "\n" \
                    "    AUC: 	" + str(np.mean(results["validation"]["auc"])) + "\n" \
                    "    sens: 	" + str(np.mean(results["validation"]["sensitivity"])) + "\n" \
                    "    spec: 	" + str(np.mean(results["validation"]["specificity"])) + "\n\n"\
                    "TEST\n" \
                    "    loss: 	" + str(np.mean(results["test"]["loss"])) + "\n" \
                    "    acc: 	" + str(np.mean(results["test"]["acc"])) + "\n" \
                    "    AUC: 	" + str(np.mean(results["test"]["auc"])) + "\n" \
                    "    sens: 	" + str(np.mean(results["test"]["sensitivity"])) + "\n" \
                    "    spec: 	" + str(np.mean(results["test"]["specificity"])) + "\n\n" \
                    "Start: 		" + str(time.strftime("%m/%d/%Y, %H:%M:%S", start_localtime)) + "\n" \
                    "End: 		" + str(time.strftime("%m/%d/%Y, %H:%M:%S", end_localtime)) + "\n" \
                    "Run time: 	" + str(round((end - start), 2)) + " seconds"

    # write to file and save
    file = open(config.output_dir + 'results_cross_val.txt', 'w')
    file.write(final_results), print(final_results)
    file.close()


def save_DL_model(model):
    """
    Saves a picture and summary of the used model in the output dir.
    """

    plot_model(model, show_shapes=True, to_file=config.output_dir + 'model.png')
    with open(config.output_dir + 'modelsummary.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()

