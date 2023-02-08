import numpy as np
import matplotlib.pyplot as plt

def plot_fnat_histograms(
        fnat_array: np.ndarray,
        fnnat_array: np.ndarray,
        savefig_path: str = None
        ):

    fig, ax = plt.subplots(1,2)
    ax[0].hist(fnat_array)
    ax[0].set_xlabel("Fnat")
    ax[1].hist(fnnat_array)
    ax[1].set_xlabel("Fnnat")

    if savefig_path is None:
        plt.show()
    else:
        plt.savefig(savefig_path, bbox_inches='tight')
    plt.close()
        
def compare_cmaps(
    cm_true: np.ndarray,
    cm_pred: np.ndarray,
    thresh: int = 8,
    dist_max: int = 25,
    pid0: str = '',
    pid1: str = '',
    suptitle: str = '',
    color: str = "Blues_r",
    vmin: float = 0,
    vmax: float = 26,
    savefig_path: str = None
):
    cm_true_bin = (cm_true < thresh).long()
    cm_pred_bin = (cm_pred < thresh).long()
    
    fig, ax = plt.subplots(1, 4, figsize=(15, 5))
    fig.suptitle(suptitle)
    
    ax[0].imshow(cm_true_bin, cmap=color)
    ax[0].set_xlabel(pid1)
    ax[0].set_ylabel(pid0)
    ax[0].set_title("True Contacts")
    
    ax[1].imshow(cm_pred_bin, cmap=color)
    ax[1].set_xlabel(pid1)
    ax[1].set_ylabel(pid0)
    ax[1].set_title("Predicted Contacts")
    
    ax[2].imshow(dist_max - cm_true, vmin=vmin, vmax=vmax, cmap=color)
    ax[2].set_xlabel(pid1)
    ax[2].set_ylabel(pid0)
    ax[2].set_title("True Distances")
    
    ax[3].imshow(dist_max - cm_pred, vmin=vmin, vmax=vmax, cmap=color)
    ax[3].set_xlabel(pid1)
    ax[3].set_ylabel(pid0)
    ax[3].set_title("Predicted Distances")

    if savefig_path is None:
        plt.show()
    else:
        plt.savefig(savefig_path, bbox_inches='tight')
    plt.close()
