import numpy as np

def calc_f_nat(cm_true: np.ndarray, cm_pred: np.ndarray):
    """
    cm_true and cm_pred should both already be binary matrices, thresholded at either 5 or 8 angstroms, or some probability threshold
    """
    n_native = cm_true.sum()
    n_preserved = (cm_true & cm_pred).sum()
    f_nat = n_preserved / n_native

    return f_nat

def calc_f_nonnat(cm_true: np.ndarray, cm_pred: np.ndarray):
    """
    cm_true and cm_pred should both already be binary matrices, thresholded at either 5 or 8 angstroms, or some probability threshold
    """
    n_nonnative = ((1 - cm_true) & cm_pred).sum()
    n_predicted = cm_pred.sum()
    if n_predicted == 0:
        return 0
    else:
        f_nonnat = n_nonnative / n_predicted
        return f_nonnat
