import numpy as np
import torch
from torchmetrics import Metric

class FNat(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = True
    
    def __init__(self, thresh: float = 8):
        super().__init__()
        self.thresh = 8
        self.add_state("metric", default = torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("n", default = torch.tensor(0), dist_reduce_fx="sum")
        
    def update(self, cm_true: torch.Tensor, cm_pred: torch.Tensor):
        assert cm_true.shape == cm_pred.shape

        self.metric += calc_f_nat(cm_true, cm_pred, thresh = self.thresh)
        self.n += 1
        
    def compute(self):
        return self.metric.float() / self.n
    
class FNonNat(Metric):
    is_differentiable = False
    higher_is_better = False
    full_state_update = True
    
    def __init__(self, thresh: float = 8):
        super().__init__()
        self.thresh = 8
        self.add_state("metric", default = torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("n", default = torch.tensor(0), dist_reduce_fx="sum")
        
    def update(self, cm_true: torch.Tensor, cm_pred: torch.Tensor):
        assert cm_true.shape == cm_pred.shape

        self.metric += calc_f_nonnat(cm_true, cm_pred, thresh = self.thresh)
        self.n += 1
        
    def compute(self):
        return self.metric.float() / self.n
    
class TopKPrecision(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = True
    
    def __init__(self, thresh: float = 8, k: int = 10):
        super().__init__()
        self.thresh = 8
        if k < 1:
            raise ValueError(f"k={k} must be at least 1")
        self.k = k
        self.add_state("metric", default = torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("n", default = torch.tensor(0), dist_reduce_fx="sum")
        
    def update(self, cm_true: torch.Tensor, cm_pred: torch.Tensor):
        assert cm_true.shape == cm_pred.shape
        
        self.metric += calc_top_k_precision(cm_true, cm_pred, k = self.k, thresh = self.thresh)
        self.n += 1
        
    def compute(self):
        return self.metric.float() / self.n
    
class TopLPrecision(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = True
    
    def __init__(self, thresh: float = 8, Ldiv: int = 10):
        super().__init__()
        self.thresh = 8
        if Ldiv < 1:
            raise ValueError(f"k={k} must be at least 1")
        self.Ldiv = Ldiv
        self.add_state("metric", default = torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("n", default = torch.tensor(0), dist_reduce_fx="sum")
        
    def update(self, cm_true: torch.Tensor, cm_pred: torch.Tensor):
        assert cm_true.shape == cm_pred.shape
        
        self.metric += calc_top_Ldiv_precision(cm_true, cm_pred, Ldiv = self.Ldiv, thresh = self.thresh)
        self.n += 1
        
    def compute(self):
        return self.metric.float() / self.n
    
def calc_f_nat(cm_true: torch.Tensor, cm_pred: torch.Tensor, thresh: float = 8):
    """
    cm_true and cm_pred should be distances in units of angstroms. Distances less than thresh will be considered positives.
    """
    cm_true_bin = (cm_true < thresh).long()
    cm_pred_bin = (cm_pred < thresh).long()
    
    n_native = torch.sum(cm_true_bin)
    n_preserved = torch.sum(cm_true_bin & cm_pred_bin)
    f_nat = n_preserved.float() / n_native

    return f_nat

def calc_f_nonnat(cm_true: torch.Tensor, cm_pred: torch.Tensor, thresh: float = 8):
    """
    cm_true and cm_pred should be distances in units of angstroms. Distances less than thresh will be considered positives.
    """
    cm_true_bin = (cm_true < thresh).long()
    cm_pred_bin = (cm_pred < thresh).long()
    
    n_nonnative = torch.sum((1 - cm_true_bin) & cm_pred_bin)
    n_predicted = torch.sum(cm_pred_bin)
    if n_predicted == 0:
        return 0
    else:
        f_nonnat = n_nonnative.float() / n_predicted
        return f_nonnat
    
def calc_top_k_precision(cm_true: torch.Tensor, cm_pred: torch.Tensor, k: int = 10, thresh: float = 8):
    """
    cm_true and cm_pred should be distances in units of angstroms. Distances less than thresh will be considered positives.
    top_k precision computes the proportion of the top_k predicted contacts which appear in the true contact map.
    """
    cm_true_bin = (cm_true < thresh).long()
    
    true_ravel = cm_true_bin.ravel()
    pred_ravel = cm_pred.ravel()
    
    if k > len(true_ravel):
        raise ValueError(f"k={k}is larger than the total number of possible contacts")
    elif k < 1:
        raise ValueError(f"k={k} must be at least 1")
    
    argpartition = np.argpartition(pred_ravel, k-1)
    
    top_k_pred = pred_ravel[argpartition][:k]
    top_k_labels = true_ravel[argpartition][:k]
    top_k_precision = torch.sum(top_k_labels).float() / k
    
    return top_k_precision

def calc_top_Ldiv_precision(cm_true: torch.Tensor, cm_pred: torch.Tensor, Ldiv = 10, thresh: float = 8):
    """
    cm_true and cm_pred should be distances in units of angstroms. Distances less than thresh will be considered positives.
    This function computes top_k_precision, but based on L/Ldiv, where L is the length of the shorter of the two chains.
    """
    shape = cm_true.shape
    L = min(cm_true.shape)
    
    if Ldiv > L:
        raise ValueError(f"Ldiv={Ldiv} is greater than the minimum sequence {L}")
    
    return calc_top_k_precision(cm_true, cm_pred, k = max(int(L / Ldiv), 1), thresh = thresh)