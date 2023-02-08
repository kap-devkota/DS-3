import numpy as np

def calc_f_nat(cm_true: np.ndarray, cm_pred: np.ndarray, thresh: float = 8):
    """
    cm_true and cm_pred should be distances in units of angstroms. Distances less than thresh will be considered positives.
    """
    cm_true_bin = (cm_true < thresh).astype(int)
    cm_pred_bin = (cm_pred < thresh).astype(int)
    
    n_native = cm_true_bin.sum()
    n_preserved = (cm_true_bin & cm_pred_bin).sum()
    f_nat = n_preserved / n_native

    return f_nat

def calc_f_nonnat(cm_true: np.ndarray, cm_pred: np.ndarray, thresh: float = 8):
    """
    cm_true and cm_pred should be distances in units of angstroms. Distances less than thresh will be considered positives.
    """
    cm_true_bin = (cm_true < thresh).astype(int)
    cm_pred_bin = (cm_pred < thresh).astype(int)
    
    n_nonnative = ((1 - cm_true_bin) & cm_pred_bin).sum()
    n_predicted = cm_pred_bin.sum()
    if n_predicted == 0:
        return 0
    else:
        f_nonnat = n_nonnative / n_predicted
        return f_nonnat
    
def calc_top_k_precision(cm_true: np.ndarray, cm_pred: np.ndarray, thresh: float = 8, k: int = 10):
    """
    cm_true and cm_pred should be distances in units of angstroms. Distances less than thresh will be considered positives.
    top_k precision computes the proportion of the top_k predicted contacts which appear in the true contact map.
    """
    raise NotImplementedError()

def calc_top_Ldiv_precision(cm_true: np.ndarray, cm_pred: np.ndarray, Ldiv = 10):
    """
    cm_true and cm_pred should be distances in units of angstroms. Distances less than thresh will be considered positives.
    This function computes top_k_precision, but based on L/Ldiv, where L is the length of the shorter of the two chains.
    """
    shape = cm_true.shape
    L = min(cm_true.shape)
    
    return calc_top_k_precision(cm_true, cm_pred, k = L / Ldiv)