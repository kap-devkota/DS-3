import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class StructCmap(nn.Module):
    def __init__(self, init_dim, project_dim, n_head_within, n_bins, ppi_window = 10, drop = 0.2, activation = "tanh", **kwargs):
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
    def __init__(self, dim, proj, nhead, drop = 0.2, **kwargs):
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
    def __init__(self, init_dim, project_dim, n_head_within, n_bins, ppi_window = 10, drop = 0.2, activation = "tanh", **kwargs):
        super(StructCmapCATT, self).__init__()
        # activations allowed
        activations = {"sigmoid" : nn.Sigmoid(), "tanh" : nn.Tanh(), "relu" : nn.ReLU()}
        self.project    = nn.Linear(init_dim, project_dim)
        self.drop1      = nn.Dropout(p = drop)
        self.activation = activations[activation]
        
        self.mha_within = nn.MultiheadAttention(project_dim, n_head_within, dropout = drop, batch_first = True)
        self.cross_att  = nn.MultiheadAttention(project_dim, n_head_within, dropout = drop, batch_first = True)
        # CrossAttention(project_dim, project_dim, n_head_within)
        
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
        
        x1_ = self.activation(x1) # batch x nseq x project_dim
        x2_ = self.activation(x2)
        
        # Cross attention
        x1 = x1 + self.cross_att(x1_, x2_, x2_)[0]
        x2 = x2 + self.cross_att(x2_, x1_, x1_)[0]
        
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
    def __init__(self, init_dim, project_dim, n_head_within, n_bins, ppi_window = 10, drop = 0.2, activation = "tanh", wtype = "tri", wsize = 5, **kwargs):
        super(WindowedStructCmapCATT, self).__init__()
        # activations allowed
        activations = {"sigmoid" : nn.Sigmoid(), "tanh" : nn.Tanh(), "relu" : nn.ReLU()}
        self.project    = nn.Linear(init_dim, project_dim)
        self.drop1      = nn.Dropout(p = drop)
        self.activation = activations[activation]
        
        self.mha_within = nn.MultiheadAttention(project_dim, n_head_within, dropout = drop, batch_first = True)
        self.cross_att  = nn.MultiheadAttention(project_dim, n_head_within, dropout = drop, batch_first = True)
        # CrossAttention(project_dim, project_dim, n_head_within)
        
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
        
        x1_ = self.activation(x1) # batch x nseq x project_dim
        x2_ = self.activation(x2)
        
        # Cross attention
        x1 = x1 + self.cross_att(x1_, x2_, x2_)[0]
        x2 = x2 + self.cross_att(x2_, x1_, x1_)[0]
        
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


class SelfCrossBlock(nn.Module):
    def __init__(self, proj_dim, n_head, activation = "gelu", dropout = 0.2, allow_cross = True, gelu_mid = False, **kwargs):
        super(SelfCrossBlock, self).__init__()
        activations = {"sigmoid" : nn.Sigmoid(), "tanh" : nn.Tanh(), "relu" : nn.ReLU(), "gelu" : nn.GELU()}
        if activation not in activations:
            self.activation = nn.Identity()
        else:
            self.activation = activations[activation]
        self.allow_cross = allow_cross
        self.mha_within = nn.MultiheadAttention(proj_dim, n_head, dropout = dropout, batch_first = True)
        self.gelu_mid = gelu_mid
        if self.allow_cross:
            if gelu_mid:
                self.gelu = nn.GELU()
            self.cross_att  = nn.MultiheadAttention(proj_dim, n_head, drop = dropout, batch_first = True)
    
    def forward(self, x1, x2):
        x1 = self.mha_within(x1, x1, x1)[0] + x1
        x2 = self.mha_within(x2, x2, x2)[0] + x2
        
        if self.allow_cross:
            if self.gelu_mid:
                x1_ = self.gelu(x1)
                x2_ = self.gelu(x2)
            else:
                x1_ = self.activation(x1)
                x2_ = self.activation(x2)
            x1 = x1 + self.cross_att(x1_, x2_, x2_)[0]
            x2 = x2 + self.cross_att(x2_, x1_, x1_)[0]
        return self.activation(x1), self.activation(x2)
    
class ConvBlock(nn.Module):
    def __init__(self, 
                 channels,
                 kernels, 
                 activation):
        super(ConvBlock, self).__init__()
        activations = {"sigmoid" : nn.Sigmoid(), "tanh" : nn.Tanh(), "relu" : nn.ReLU(), "gelu" : nn.GELU()}
        self.activation = activations[activation]
        self.cmodules  = nn.ModuleList()
        prev_channel  = channels[0]
        channels      = channels[1:]
        for c, k in zip(channels, kernels):
            self.cmodules.append(nn.Conv2d(prev_channel, c, (k, k), padding = 'same'))
            prev_channel = c
        return 
    
    def forward(self, cm):
        output = cm
        for i, mod in enumerate(self.cmodules):
            output = mod(output)
        return output
    
class WindowedStackedStructCmapCATT(nn.Module):
    def __init__(self, 
                 init_dim, 
                 project_dim = 100, 
                 n_head_within = 5, 
                 n_crossblock = 2, 
                 n_bins = 25, 
                 ppi_window = 10, 
                 drop = 0.2, 
                 activation = "sigmoid", 
                 wtype = "tri", 
                 wsize = 3,
                 allow_cross = True, 
                 conv_channels = [],
                 kernels = [],
                 **kwargs):
        super(WindowedStackedStructCmapCATT, self).__init__()
        # activations allowed
        activations = {"sigmoid" : nn.Sigmoid(), "tanh" : nn.Tanh(), "relu" : nn.ReLU(), "gelu" : nn.GELU()}
        self.project    = nn.Linear(init_dim, project_dim)
        self.drop1      = nn.Dropout(p = drop)
        
        if activation not in activations:
            self.activation = nn.Identity()
        else:
            self.activation = activations[activation]
        
        self.blockmodules = nn.ModuleList()
        for i in range(n_crossblock):
            self.blockmodules.append(SelfCrossBlock(project_dim, n_head_within, activation = activation, dropout = drop, allow_cross = allow_cross))
        
        
        # Convolution block
        cv_channels = conv_channels + [n_bins]
        self.W = nn.Parameter(torch.randn(cv_channels[0], project_dim, (project_dim // 2), dtype = torch.float32))
        
        self.convlayers = False
        if len(cv_channels) > 1:
            assert len(cv_channels) == (len(kernels) + 1)
            self.convblock = ConvBlock(cv_channels, kernels, activation)
            self.convlayers = True
            
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
        
        for i, mod in enumerate(self.blockmodules):
            x1, x2 = mod(x1, x2)
                                
        x1 = torch.matmul(x1.unsqueeze(1), self.W) # X => batch x 1 x nseq1 x proj_dim W => nbin x proj_dim x out_proj_dim # batch x nbin x nseq x out_proj_dim
        x2 = torch.matmul(x2.unsqueeze(1), self.W) 
        
        x1 = self.drop1(self.activation(x1))
        x2 = self.drop1(self.activation(x2)) # batch x nbin x n_seq x pr_dim
        cm  = torch.matmul(x1, torch.transpose(x2, 2, 3))  # batch x n_bin x n_seq1 x pr_dim times batch x n_bin x pr_dim x n_seq2 => batch x n_bin x n_seq1 x n_seq2
        
        ## Window here
        cm = F.pad(cm, (0, 0, 0, 0, self.smoothpad, self.smoothpad)).unfold(1, self.smoothsize, 1) #batch x n_bin x n_seq1 x n_seq2 x window
        cm = torch.sum(cm * self.smooth.view(1, 1, 1, 1, -1), dim = 4)
        
        if self.convlayers:
            cm = self.convblock(cm) 
        
        cm1  = torch.sum(cm * self.agg, axis = 1) # aggregation => batch x n_seq1 x n_seq2
        cm1  = self.activation(cm1)
        
        windows = F.unfold(cm1.unsqueeze(1), kernel_size = (self.ppi_window, self.ppi_window)) # => batch x (ppi_window x ppi_window) x no_windows
        wsum = torch.sum(windows, axis = 2)
        
        wsum = self.drop1(self.activation(wsum))
        pp_prob = torch.sigmoid(torch.matmul(wsum, self.L) + self.b)
        # cm[:, -1, :, :] /= pp_prob # If very low prob, then very large distance
        # cm[:, :-1, :, :] /= (1-pp_prob) # If very high prob, then very low distance
        return cm, pp_prob


class SelfCrossBlock2(nn.Module):
    def __init__(self, proj_dim, n_head, activation = "gelu", dropout = 0.2, allow_norm = True, **kwargs):
        super(SelfCrossBlock2, self).__init__()
        activations = {"sigmoid" : nn.Sigmoid(), "tanh" : nn.Tanh(), "relu" : nn.ReLU(), "gelu" : nn.GELU()}
        if activation not in activations:
            self.activation = nn.Identity()
        else:
            self.activation = activations[activation]
        self.mha_within = nn.MultiheadAttention(proj_dim, n_head, dropout = dropout, batch_first = True)
        self.cross_att  = nn.MultiheadAttention(proj_dim, n_head, dropout = dropout, batch_first = True)
        if allow_norm:
            self.lnorm = nn.LayerNorm(proj_dim)
        self.allow_norm = allow_norm
        
    """
    def forward(self, x1, x2, x3, x4):
        x1 = self.mha_within(x1, x1, x1)[0] + x1
        x2 = self.mha_within(x2, x2, x2)[0] + x2
        
        x3_ = self.cross_att(x3, x4, x2)[0] + x3 
        x4_ = self.cross_att(x4, x3, x1)[0] + x4 
        return self.activation(x1), self.activation(x2), self.activation(x3_), self.activation(x4_)
    """
    def forward(self, x1, x2):
        x1 = self.mha_within(x1, x1, x1)[0]
        x2 = self.mha_within(x2, x2, x2)[0]
        
        x1_ = self.cross_att(x1, x2, x2)[0]
        x2_ = self.cross_att(x2, x1, x1)[0]
        
        x1_ = self.activation(x1_)
        x2_ = self.activation(x2_)
        
        if self.allow_norm:
            x1_ = self.lnorm(x1_)
            x2_ = self.lnorm(x2_)
        return x1_, x2_
    


    
    
class PosEncode(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PosEncode, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[0, :x.size(0), :]
        return self.dropout(x)
 
class WindowedStackedStructCmapCATT2(nn.Module):
    def __init__(self, 
                 init_dim, 
                 project_dim = 100, 
                 n_head_within = 5, 
                 n_crossblock = 2, 
                 n_bins = 25, 
                 ppi_window = 10, 
                 drop = 0.2, 
                 activation = "sigmoid", 
                 wtype = "tri", 
                 wsize = 3,
                 conv_channels = [],
                 kernels = [],
                 **kwargs):
        super(WindowedStackedStructCmapCATT2, self).__init__()
        # activations allowed
        activations = {"sigmoid" : nn.Sigmoid(), "tanh" : nn.Tanh(), "relu" : nn.ReLU(), "gelu" : nn.GELU()}
        self.project    = nn.Linear(init_dim, project_dim)
        self.drop1      = nn.Dropout(p = drop)
        self.posenc     = PosEncode(project_dim)
        if activation not in activations:
            self.activation = nn.Identity()
        else:
            self.activation = activations[activation]
        
        self.blockmodules = nn.ModuleList()
        for i in range(n_crossblock):
            self.blockmodules.append(SelfCrossBlock2(project_dim, n_head_within, activation = activation, dropout = drop))
        
        
        # Convolution block
        cv_channels = conv_channels + [n_bins]

        self.W = nn.Parameter(torch.randn(cv_channels[0], project_dim, (project_dim // 2), dtype = torch.float32))
        # self.W = nn.Parameter(torch.randn(cv_channels[0], 2 * project_dim, (project_dim // 2), dtype = torch.float32))
        
        self.convlayers = False
        if len(cv_channels) > 1:
            assert len(cv_channels) == (len(kernels) + 1)
            self.convblock = ConvBlock(cv_channels, kernels, activation)
            self.convlayers = True
            
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
        
        # position encoding
        x1 = self.posenc(x1)
        x2 = self.posenc(x2)
        
#         x3 = x1
#         x4 = x2
#         for i, mod in enumerate(self.blockmodules):
#             x1, x2, x3, x4 = mod(x1, x2, x3, x4)

        
#         x1 = torch.cat([x1, x3], dim = -1)
#         x2 = torch.cat([x2, x4], dim = -1)
                     
        for i, mode in enumerate(self.blockmodules):
            x3, x4 = mode(x1, x2)
            x1 = x1 + x3
            x2 = x2 + x4
        
        x1 = self.activation(x1)
        x2 = self.activation(x2)
                       
        x1 = torch.matmul(x1.unsqueeze(1), self.W) # X => batch x 1 x nseq1 x proj_dim W => nbin x proj_dim x out_proj_dim # batch x nbin x nseq x out_proj_dim
        x2 = torch.matmul(x2.unsqueeze(1), self.W) 
        
        x1 = self.drop1(self.activation(x1))
        x2 = self.drop1(self.activation(x2)) # batch x nbin x n_seq x pr_dim
        cm  = torch.matmul(x1, torch.transpose(x2, 2, 3))  # batch x n_bin x n_seq1 x pr_dim times batch x n_bin x pr_dim x n_seq2 => batch x n_bin x n_seq1 x n_seq2  
        if self.convlayers:
            cm = self.convblock(cm) 
        
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
    
    
#####################################    
######## Full-on Transformer ########
#####################################
class SelfCrossBlock3(nn.Module):
    def __init__(self, proj_dim, n_head, activation = "gelu", dropout = 0.2, n_transformer = 2, **kwargs):
        super(SelfCrossBlock3, self).__init__()
        activations = {"sigmoid" : torch.sigmoid, "tanh" : torch.tanh, "relu" : torch.relu, "gelu" : F.gelu}
        if activation not in activations:
            self.activation = nn.Identity()
        else:
            self.activation = activations[activation]
        
        # within encoder
        within_encoder = nn.TransformerEncoderLayer(d_model= proj_dim, nhead = n_head, dim_feedforward = proj_dim, dropout = dropout, batch_first = True) 
        self.mha_within = nn.TransformerEncoder(within_encoder, num_layers = n_transformer)
        
        # cross decoder
        cross_decoder = nn.TransformerDecoderLayer(d_model= proj_dim, nhead = n_head, dim_feedforward = proj_dim, dropout = dropout, batch_first = True)
        self.cross_att  = nn.TransformerDecoder(cross_decoder, num_layers = n_transformer)
    
    def forward(self, x1, x2):
        x1 = self.mha_within(x1)
        x2 = self.mha_within(x2)
        
        x1_ = self.cross_att(x1, x2)
        x2_ = self.cross_att(x2, x1)
        
        return self.activation(x1_), self.activation(x2_)
    
 
class WindowedStackedStructCmapCATT3(nn.Module):
    def __init__(self, 
                 init_dim, 
                 project_dim = 100, 
                 n_head_within = 5, 
                 n_crossblock = 1, 
                 n_crossblock_layers=2,
                 n_bins = 25, 
                 ppi_window = 10, 
                 drop = 0.2, 
                 activation = "sigmoid", 
                 wtype = "tri", 
                 wsize = 3,
                 conv_channels = [],
                 kernels = [],
                 **kwargs):
        super(WindowedStackedStructCmapCATT3, self).__init__()
        # activations allowed
        activations = {"sigmoid" : nn.Sigmoid(), "tanh" : nn.Tanh(), "relu" : nn.ReLU(), "gelu" : nn.GELU()}
        self.project    = nn.Linear(init_dim, project_dim)
        self.drop1      = nn.Dropout(p = drop)
        self.posenc     = PosEncode(project_dim)
        if activation not in activations:
            self.activation = nn.Identity()
        else:
            self.activation = activations[activation]
        
        self.blockmodules = nn.ModuleList()
        for i in range(n_crossblock):
            self.blockmodules.append(SelfCrossBlock3(project_dim, n_head_within, activation = activation, dropout = drop, n_transformer = n_crossblock_layers))
        
        
        # Convolution block
        cv_channels = conv_channels + [n_bins]
        self.W = nn.Parameter(torch.randn(cv_channels[0], project_dim, (project_dim // 2), dtype = torch.float32))
        
        self.convlayers = False
        if len(cv_channels) > 1:
            assert len(cv_channels) == (len(kernels) + 1)
            self.convblock = ConvBlock(cv_channels, kernels, activation)
            self.convlayers = True
            
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
        
        # position encoding
        x1 = self.posenc(x1)
        x2 = self.posenc(x2)
        
                     
        for i, mode in enumerate(self.blockmodules):
            x3, x4 = mode(x1, x2)
            x1 = x1 + x3
            x2 = x2 + x4
        
        x1 = self.activation(x1)
        x2 = self.activation(x2)
                       
        x1 = torch.matmul(x1.unsqueeze(1), self.W) # X => batch x 1 x nseq1 x proj_dim W => nbin x proj_dim x out_proj_dim # batch x nbin x nseq x out_proj_dim
        x2 = torch.matmul(x2.unsqueeze(1), self.W) 
        
        x1 = self.drop1(self.activation(x1))
        x2 = self.drop1(self.activation(x2)) # batch x nbin x n_seq x pr_dim
        cm  = torch.matmul(x1, torch.transpose(x2, 2, 3))  # batch x n_bin x n_seq1 x pr_dim times batch x n_bin x pr_dim x n_seq2 => batch x n_bin x n_seq1 x n_seq2  
        if self.convlayers:
            cm = self.convblock(cm) 
        
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

#################################################
#################################################
#################################################
#################################################
class WindowedStackedStructCmapCATT4(nn.Module):
    def __init__(self, 
                 init_dim, 
                 project_ff_dims = [256, 100], 
                 n_head_within = 5, 
                 n_crossblock = 1, 
                 n_crossblock_layers=2,
                 n_bins = 25, 
                 drop = 0.2, 
                 activation = "sigmoid", 
                 wtype = "tri", 
                 wsize = 3,
                 conv_channels = [],
                 kernels = [],
                 **kwargs):
        super(WindowedStackedStructCmapCATT4, self).__init__()
        # activations allowed
        activations = {"sigmoid" : torch.sigmoid, "tanh" : torch.tanh, "relu" : torch.relu, "gelu" : F.gelu}
        
        project = []
        prev_dim = init_dim
        for i in project_ff_dims:
            project += [nn.Linear(prev_dim, i), nn.ReLU()]
            prev_dim = i
        
        project_dim = project_ff_dims[-1]
        
        self.project = nn.Sequential(*project)
        
        self.drop1      = nn.Dropout(p = drop)
        self.posenc     = PosEncode(project_dim)
        if activation not in activations:
            self.activation = nn.Identity()
        else:
            self.activation = activations[activation]
        
        self.blockmodules = nn.ModuleList()
        for i in range(n_crossblock):
            self.blockmodules.append(SelfCrossBlock3(project_dim, n_head_within, activation = activation, dropout = drop, n_transformer = n_crossblock_layers))
        
        # Convolution block
        cv_channels = conv_channels + [n_bins]
        self.W = nn.Parameter(torch.randn(cv_channels[0], project_dim, (project_dim // 2), dtype = torch.float32))
        
        self.convlayers = False
        if len(cv_channels) > 1:
            assert len(cv_channels) == (len(kernels) + 1)
            self.convblock = ConvBlock(cv_channels, kernels, activation)
            self.convlayers = True
        
        self.skip_connection = False if "skip_connection" not in kwargs else kwargs["skip_connection"]
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
        
        # position encoding
        x1 = self.posenc(x1)
        x2 = self.posenc(x2)
        
        # For skip connection
        x1_ = x1
        x2_ = x2
        for i, mode in enumerate(self.blockmodules):
            x3, x4 = mode(x1, x2)
            x1 = x1 + x3
            x2 = x2 + x4
            
        # Skip connection
        if self.skip_connection:
            x1 = x1 + x1_
            x2 = x2 + x2_
        
        x1 = self.activation(x1)
        x2 = self.activation(x2)
                       
        x1 = torch.matmul(x1.unsqueeze(1), self.W) # X => batch x 1 x nseq1 x proj_dim W => nbin x proj_dim x out_proj_dim # batch x nbin x nseq x out_proj_dim
        x2 = torch.matmul(x2.unsqueeze(1), self.W) 
        
        x1 = self.drop1(self.activation(x1))
        x2 = self.drop1(self.activation(x2)) # batch x nbin x n_seq x pr_dim
        cm  = torch.matmul(x1, torch.transpose(x2, 2, 3))  # batch x n_bin x n_seq1 x pr_dim times batch x n_bin x pr_dim x n_seq2 => batch x n_bin x n_seq1 x n_seq2  
        if self.convlayers:
            cm = self.convblock(cm) 
        
        ## Window here
        cm = F.pad(cm, (0, 0, 0, 0, self.smoothpad, self.smoothpad)).unfold(1, self.smoothsize, 1) #batch x n_bin x n_seq1 x n_seq2 x window
        cm = torch.sum(cm * self.smooth.view(1, 1, 1, 1, -1), dim = 4)
        
        return cm
