import numpy as np
from statsmodels.stats.contingency_tables import mcnemar


def main():
    """
    This script can perform a statistical McNemar test based on the correct and miss classified subjects of two
    methods. The McNemar Chi-square test is used to analyse statistical differences between the classifiers based on
    accuracy. The McNemar test is a non-parametric test used on nominal data that addresses the subjects both methods
    disagree on. The test is applied to 2x2 contingency tables with a dichotomous trait with matched pairs of
    subjects and determined whether the row and column marginal frequencies were equal. This code is a python
    implementation of the original Matlab code adapted from http://www. mathworks.com/matlabcentral/fileexchange/15472
    """

    save_path = "/path/to/save/stats/"
    task = "AD"                 # AD / MCI
    folds = 10                  # amount of folds to perform the stats over
    tv = "test"                 # train / test / val

    results_method1 = "/path/to/results/dir/of/method1/k"
    results_method2 = "/path/to/results/dir/of/method2/k"

    # create predictions for both methods
    m1_ypred = []
    m2_ypred = []
    ytrue = []

    # create confusion matrix
    conf = {"both": 0, "M1": 0, "M2": 0, "none": 0}

    # loop over all folds in the output dir
    for i in range(0, folds):

        # get prediction of method 1
        m1_path = results_method1 + str(i) + "/"
        print(m1_path)
        m1_y_pred = np.load(f"{m1_path}{tv}_y_pred.npy")
        m1_y_true = np.load(f"{m1_path}{tv}_y_true.npy")
        m1_ids = np.load(f"{m1_path}{tv}_IDs.npy")

        # get predictions of method 2
        m2_path = results_method2 + str(i) + "/"
        print(m2_path)
        m2_y_pred = np.load(f"{m2_path}{tv}_y_pred.npy")
        m2_y_true = np.load(f"{m2_path}{tv}_y_true.npy")
        m2_ids = np.load(f"{m2_path}{tv}_IDs.npy")

        # compare the order of IDs (should be the same for both methods)
        if set(m2_ids) != set(m1_ids):
            print("WARNING: IDs are not the same")
        else:
            print("IDs are the same")

        # compare true labels (should be the same for both methods)
        if set(m1_y_true) != set(m2_y_true):
            print("WARNING: true labels are not the same\n")
        else:
            print("True labels are the same\n")

        # extend predictions to create one list for all folds
        m1_ypred.extend(m1_y_pred)
        m2_ypred.extend(m2_y_pred)
        ytrue.extend(m1_y_true)

        # calculate correct classified subjects per method
        m1_corr = m1_y_pred - m1_y_true
        m2_corr = m2_y_pred - m2_y_true

        # count correct and miss classified subjects
        m1_corr_subs, m1_miss_subs = count_classifications(m1_corr, m1_ids)
        m2_corr_subs, m2_miss_subs = count_classifications(m2_corr, m2_ids)

        # add to confusion matrix
        conf["both"] = conf["both"] + len(list(set(m1_corr_subs).intersection(m2_corr_subs)))
        conf["none"] = conf["none"] + len(list(set(m1_miss_subs).intersection(m2_miss_subs)))
        conf["GM"] = conf["GM"] + len(set(m1_corr_subs) - set(m2_corr_subs))
        conf["T1"] = conf["T1"] + len(set(m2_corr_subs) - set(m1_corr_subs))

    # report total results
    np.save(f"{save_path}{task}_conf_matrix.npy", conf)
    print("Both: " + str(conf["both"]))
    print("None: " + str(conf["none"]))
    print("Method 1: " + str(conf["M1"]))
    print("Method 2: " + str(conf["M2"]))

    # perform McNemar test on confusion table
    table = [[conf["both"], conf["M2"]], [conf["M1"], conf["none"]]]
    result = mcnemar(table, exact=False, correction=True)

    # summarize the finding
    print('\nstatistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))

    # interpret the p-value
    alpha = 0.05
    if result.pvalue > alpha:
        print('Same proportions of errors (fail to reject H0)')
    else:
        print('Different proportions of errors (reject H0)')

    np.save(f"{save_path}mcnemar.npy", result)

    print('\nend')


def count_classifications(corr, ids):
    """
    Counts the amount of correct and miss classified subjects.
    """

    corr_subs = []
    miss_subs = []

    for i, c in enumerate(corr):
        if c == 0:
            corr_subs.append(ids[i])
        else:
            miss_subs.append(ids[i])

    return corr_subs, miss_subs


if __name__ == '__main__':
    main()
