import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class StructCmap(nn.Module):
    def __init__(self, init_dim, project_dim, n_head_within, n_bins, ppi_window = 10, drop = 0.2, activation = "tanh"):
        super(StructCmap, self).__init__()
        # activations allowed
        activations = {"sigmoid" : nn.Sigmoid(), "tanh" : nn.Tanh(), "relu" : nn.ReLU()}
        self.project = nn.Linear(init_dim, project_dim)
        self.drop1   = nn.Dropout(p = drop)
        self.activation = activations[activation]
        
        self.mha_within = nn.MultiheadAttention(project_dim, n_head_within, dropout = drop)
        self.W = nn.Parameter(torch.randn(n_bins, project_dim, (project_dim // 2), dtype = torch.float32))
        self.P = nn.Parameter(torch.randn(n_bins, ppi_window, ppi_window, dtype = torch.float32))
        self.L = nn.Parameter(torch.randn(ppi_window * ppi_window, 1, dtype = torch.float32))
        self.ppi_window = ppi_window
        self.agg = nn.Parameter(torch.randn(1, n_bins, 1, 1, dtype = torch.float32))
        self.b = nn.Parameter(torch.tensor(0, dtype = torch.float32))
    
    def forward(self, x1, x2):
        x1 = self.project(x1)
        x2 = self.project(x2)
        
        x1 = self.drop1(x1)
        x2 = self.drop1(x2)
        
        x1 = self.mha_within(x1, x1, x1)[0] + x1
        x2 = self.mha_within(x2, x2, x2)[0] + x2 # batch x nseq2 x project_dim
        
        x1 = self.activation(x1) # batch x nseq x project_dim
        x2 = self.activation(x2)
        
        x1 = torch.matmul(x1.unsqueeze(1), self.W) # batch x nbin x nseq x nproj
        x2 = torch.matmul(x2.unsqueeze(1), self.W) 
        
        x1 = self.drop1(self.activation(x1))
        x2 = self.drop1(self.activation(x2)) # batch x nbin x n_seq x pr_dim
        cm  = torch.matmul(x1, torch.transpose(x2, 2, 3))  # batch x n_bin x n_seq1 x pr_dim times batch x n_bin x pr_dim x n_seq2 => batch x n_bin x n_seq1 x n_seq2
        cm1  = torch.sum(cm * self.agg, axis = 1) # aggregation => batch x n_seq1 x n_seq2
        cm1  = self.activation(cm1)
        
        windows = F.unfold(cm1.unsqueeze(1), kernel_size = (self.ppi_window, self.ppi_window)) # => batch x (ppi_window x ppi_window) x no_windows
        wsum = torch.sum(windows, axis = 2)
        
        wsum = self.drop1(self.activation(wsum))
        pp_prob = torch.sigmoid(torch.matmul(wsum, self.L) + self.b)
        # cm[:, -1, :, :] /= pp_prob # If very low prob, then very large distance
        # cm[:, :-1, :, :] /= (1-pp_prob) # If very high prob, then very low distance
        return cm, pp_prob

class CrossAttention(nn.Module):
    def __init__(self, dim, proj, nhead, drop = 0.2):
        super(CrossAttention, self).__init__()
        self.Wq = nn.Parameter(torch.randn(nhead, dim, proj), requires_grad = True)
        self.Wk = nn.Parameter(torch.randn(nhead, dim, proj), requires_grad = True)
        self.Wv = nn.Parameter(torch.randn(nhead, dim, proj), requires_grad = True)
        self.Wo = nn.Parameter(torch.randn(nhead * proj, dim), requires_grad = True)
        self.dim = dim
        self.proj = proj
        self.drop = nn.Dropout(p=drop)
        self.nhead = nhead
        return
    
    def forward(self, q, k, v):
        """
        q => batch x seq1 x dim
        k => batch x seq2 x dim
        v => batch x seq2 x dim
        """
        q = torch.matmul(q.unsqueeze(1), self.Wq) # batch x nhead x seq1 x proj
        k = torch.matmul(k.unsqueeze(1), self.Wk) # batch x nhead x seq2 x proj
        v = torch.matmul(v.unsqueeze(1), self.Wv) # batch x nhead x seq2 x proj
        
        att = torch.softmax(torch.matmul(q, torch.transpose(k, 2, 3)) / np.sqrt(self.proj), dim = -1) # batch x nhead x seq1 x seq2
        out = torch.matmul(att, v) # batch x nhead x seq1 x proj
        out = torch.cat(out.unbind(1), dim = -1) # batch x seq1 x (proj x nhead)
        
        out = torch.matmul(out, self.Wo)
        out = self.drop(out)
        
        return out
        

class StructCmapCATT(nn.Module):
    def __init__(self, init_dim, project_dim, n_head_within, n_bins, ppi_window = 10, drop = 0.2, activation = "tanh"):
        super(StructCmapCATT, self).__init__()
        # activations allowed
        activations = {"sigmoid" : nn.Sigmoid(), "tanh" : nn.Tanh(), "relu" : nn.ReLU()}
        self.project    = nn.Linear(init_dim, project_dim)
        self.drop1      = nn.Dropout(p = drop)
        self.activation = activations[activation]
        
        self.mha_within = nn.MultiheadAttention(project_dim, n_head_within, dropout = drop)
        self.cross_att  = CrossAttention(project_dim, project_dim, n_head_within)
        
        self.W = nn.Parameter(torch.randn(n_bins, project_dim, (project_dim // 2), dtype = torch.float32))
        self.P = nn.Parameter(torch.randn(n_bins, ppi_window, ppi_window, dtype = torch.float32))
        self.L = nn.Parameter(torch.randn(ppi_window * ppi_window, 1, dtype = torch.float32))
        
        self.ppi_window = ppi_window
        self.agg = nn.Parameter(torch.randn(1, n_bins, 1, 1, dtype = torch.float32))
        self.b = nn.Parameter(torch.tensor(0, dtype = torch.float32))
    
    def forward(self, x1, x2):
        x1 = self.project(x1)
        x2 = self.project(x2)
        
        x1 = self.drop1(x1)
        x2 = self.drop1(x2)
        
        x1 = self.mha_within(x1, x1, x1)[0] + x1
        x2 = self.mha_within(x2, x2, x2)[0] + x2 # batch x nseq2 x project_dim
        
        x1 = self.activation(x1) # batch x nseq x project_dim
        x2 = self.activation(x2)
        
        # Cross attention
        x1 = x1 + self.cross_att(x1, x2, x2)
        x2 = x2 + self.cross_att(x2, x1, x1)
        
        x1 = self.activation(x1) # batch x nseq1 x project_dim => 
        x2 = self.activation(x2) # batch x nseq2 x project_dim
        
        x1 = torch.matmul(x1.unsqueeze(1), self.W) # X => batch x 1 x nseq1 x proj_dim W => nbin x proj_dim x out_proj_dim # batch x nbin x nseq x out_proj_dim
        x2 = torch.matmul(x2.unsqueeze(1), self.W) 
        
        x1 = self.drop1(self.activation(x1))
        x2 = self.drop1(self.activation(x2)) # batch x nbin x n_seq x pr_dim
        cm  = torch.matmul(x1, torch.transpose(x2, 2, 3))  # batch x n_bin x n_seq1 x pr_dim times batch x n_bin x pr_dim x n_seq2 => batch x n_bin x n_seq1 x n_seq2
        cm1  = torch.sum(cm * self.agg, axis = 1) # aggregation => batch x n_seq1 x n_seq2
        cm1  = self.activation(cm1)
        
        windows = F.unfold(cm1.unsqueeze(1), kernel_size = (self.ppi_window, self.ppi_window)) # => batch x (ppi_window x ppi_window) x no_windows
        wsum = torch.sum(windows, axis = 2)
        
        wsum = self.drop1(self.activation(wsum))
        pp_prob = torch.sigmoid(torch.matmul(wsum, self.L) + self.b)
        # cm[:, -1, :, :] /= pp_prob # If very low prob, then very large distance
        # cm[:, :-1, :, :] /= (1-pp_prob) # If very high prob, then very low distance
        return cm, pp_prob


class WindowedStructCmapCATT(nn.Module):
    def __init__(self, init_dim, project_dim, n_head_within, n_bins, ppi_window = 10, drop = 0.2, activation = "tanh", wtype = "tri", wsize = 5):
        super(WindowedStructCmapCATT, self).__init__()
        # activations allowed
        activations = {"sigmoid" : nn.Sigmoid(), "tanh" : nn.Tanh(), "relu" : nn.ReLU()}
        self.project    = nn.Linear(init_dim, project_dim)
        self.drop1      = nn.Dropout(p = drop)
        self.activation = activations[activation]
        
        self.mha_within = nn.MultiheadAttention(project_dim, n_head_within, dropout = drop)
        self.cross_att  = CrossAttention(project_dim, project_dim, n_head_within)
        
        self.W = nn.Parameter(torch.randn(n_bins, project_dim, (project_dim // 2), dtype = torch.float32))
        self.P = nn.Parameter(torch.randn(n_bins, ppi_window, ppi_window, dtype = torch.float32))
        self.L = nn.Parameter(torch.randn(ppi_window * ppi_window, 1, dtype = torch.float32))
        
        self.ppi_window = ppi_window
        self.agg = nn.Parameter(torch.randn(1, n_bins, 1, 1, dtype = torch.float32))
        self.b = nn.Parameter(torch.tensor(0, dtype = torch.float32))
        
        assert wsize % 2 == 1

        self.smoothpad = wsize // 2
        self.smoothsize = wsize
        if wtype == "rect":
            self.smooth = nn.Parameter(torch.ones(wsize) / wsize, requires_grad = False)
        elif wtype == "tri":
            smooth = torch.concat([torch.linspace(0, 1, wsize // 2 + 2)[1:], torch.linspace(1, 0, wsize //2 +2)[1:-1]])
            smooth = smooth / torch.sum(smooth)
            self.smooth = nn.Parameter(smooth, requires_grad = False)
                                       
    def forward(self, x1, x2):
        x1 = self.project(x1)
        x2 = self.project(x2)
        
        x1 = self.drop1(x1)
        x2 = self.drop1(x2)
        
        x1 = self.mha_within(x1, x1, x1)[0] + x1
        x2 = self.mha_within(x2, x2, x2)[0] + x2 # batch x nseq2 x project_dim
        
        x1 = self.activation(x1) # batch x nseq x project_dim
        x2 = self.activation(x2)
        
        # Cross attention
        x1 = x1 + self.cross_att(x1, x2, x2)
        x2 = x2 + self.cross_att(x2, x1, x1)
        
        x1 = self.activation(x1) # batch x nseq1 x project_dim => 
        x2 = self.activation(x2) # batch x nseq2 x project_dim
        
        x1 = torch.matmul(x1.unsqueeze(1), self.W) # X => batch x 1 x nseq1 x proj_dim W => nbin x proj_dim x out_proj_dim # batch x nbin x nseq x out_proj_dim
        x2 = torch.matmul(x2.unsqueeze(1), self.W) 
        
        x1 = self.drop1(self.activation(x1))
        x2 = self.drop1(self.activation(x2)) # batch x nbin x n_seq x pr_dim
        cm  = torch.matmul(x1, torch.transpose(x2, 2, 3))  # batch x n_bin x n_seq1 x pr_dim times batch x n_bin x pr_dim x n_seq2 => batch x n_bin x n_seq1 x n_seq2
        
        ## 0.5 0.2 0.1 0.1 0.1 => 3
        ## 0 0.5 0.2 0.1 0.1 0.1 0 
        ## 0.7/3 0.8/3 0.4/3 0.1 0.2/3 
        
        
        ## 1 1 1
        ## 0.5 1 0.5 
        ## 0.33 0.66 1 0.66 0.33 
        ## Window here
        cm = F.pad(cm, (0, 0, 0, 0, self.smoothpad, self.smoothpad)).unfold(1, self.smoothsize, 1) #batch x n_bin x n_seq1 x n_seq2 x window
        cm = torch.sum(cm * self.smooth.view(1, 1, 1, 1, -1), dim = 4)
        
        cm1  = torch.sum(cm * self.agg, axis = 1) # aggregation => batch x n_seq1 x n_seq2
        cm1  = self.activation(cm1)
        
        windows = F.unfold(cm1.unsqueeze(1), kernel_size = (self.ppi_window, self.ppi_window)) # => batch x (ppi_window x ppi_window) x no_windows
        wsum = torch.sum(windows, axis = 2)
        
        wsum = self.drop1(self.activation(wsum))
        pp_prob = torch.sigmoid(torch.matmul(wsum, self.L) + self.b)
        # cm[:, -1, :, :] /= pp_prob # If very low prob, then very large distance
        # cm[:, :-1, :, :] /= (1-pp_prob) # If very high prob, then very low distance
        return cm, pp_prob
    