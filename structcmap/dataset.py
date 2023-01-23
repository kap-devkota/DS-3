from torch.utils.data import Dataset
import h5py
import pandas as pd
import torch

class PairData(Dataset):
    def __init__(self, cmap_loc, lang_loc, ppi_tsv, no_bins = None):
        self.cmap = h5py.File(cmap_loc, "r")
        self.lang = h5py.File(lang_loc, "r")
        self.no_bins = no_bins
        self.dppi = pd.read_csv(ppi_tsv, sep = "\t", header = None)
    def __len__(self):
        return len(self.dppi)
    def __getitem__(self, idx):
        p, q, sc = self.dppi.iloc[idx, :].values
        ids = f"{p}x{q}"
        Xp = torch.tensor(self.lang[p][:], dtype = torch.float32).squeeze()
        Xq = torch.tensor(self.lang[q][:], dtype = torch.float32).squeeze()
        cm = torch.tensor(self.cmap[ids][:], dtype = torch.float32).squeeze()
        
        if self.no_bins != None:
            cm = (cm / (26) * self.no_bins).long() # value in cm always < number of bins
        
        return torch.tensor(1, dtype = torch.float32), Xp, Xq, cm