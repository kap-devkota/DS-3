import numpy as np
import matplotlib.pyplot as plt

def compare_cmap(
        cm_true: np.ndarray,
        cm_pred: np.ndarray,
        pid0: str = '',
        pid1: str = '',
        color: str = "Blues_r",
        vmin: float = 0,
        vmax: float = 0.9,
        cbar: bool = True,
        savefig_path: str = None
        ):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    im0 = ax[0].imshow(cm_true, cmap=color, vmin=vmin, vmax=vmax)
    ax[0].set_title('True')
    ax[0].set_xlabel(pid1)
    ax[0].set_ylabel(pid0)
    im1 = ax[1].imshow(cm_pred, cmap=color)#, vmin=vmin, vmax=vmax)
    ax[1].set_title('Predicted')
    ax[1].set_xlabel(pid1)
    ax[1].set_ylabel(pid0)

    if cbar:
        plt.colorbar(im0, ax=ax[0], orientation = 'horizontal')
        plt.colorbar(im1, ax=ax[1], orientation = 'horizontal')
    if savefig_path is None:
        plt.show()
    else:
        plt.savefig(savefig_path, bbox_inches='tight')
