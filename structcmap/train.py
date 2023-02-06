import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import h5py
import argparse
from model import StructCmap, StructCmapCATT, WindowedStructCmapCATT, WindowedStackedStructCmapCATT, WindowedStackedStructCmapCATT2, WindowedStackedStructCmapCATT3
from dataset import PairData
import os
import matplotlib.pyplot as plt
import re
from metrics import calc_f_nat, calc_f_nonnat
from plotting import plot_losses

from omegaconf import OmegaConf
import wandb


"""
TODO:

SKIP CONNECTION
CHANGE LINEAR TO FF and add activation
WEIGHT DECAY AND Stronger dropout
LayerNorm
"""


################
### BB-DEBUG ###
################

# python train.py --train ../data/pairs/human_cm_dtrain.tsv --test ../data/pairs/human_cm_dtrain.tsv --cmap ../data/emb/cmap_d_emb.h5 --cmap_lang ../data/emb/cmap_d_lang_emb.h5 --device 3 --output_prefix ../outputs/iter_1/op_

#iter=83;mtype=-1;dev=7;odir=../outputs/iter_${iter}-bb-debug-tformer; if [ ! -d ${odir} ]; then mkdir ${odir}; cp train.py model.py $odir/; fi; CMD="python train.py --train ../data/pairs/human_cm_dtrain.tsv --test ../data/pairs/esm_test.tsv --cmap ../../D-SCRIPT/data/embeddings/cmap-latest.h5 --cmap_lang ../data/emb/lynnemb/new_cmap_embed  --device ${dev} --output_prefix ${odir}/op_ --input_dim 6165 --no_bins 25 --lrate 1e-3 --no_epoch 25 --activation sigmoid --model_type $mtype --cross_block 1 --conv_channels 45 --conv_kernels 5 --iter_dir ${odir}"; sed -Ei "1 i # Trained with command: $CMD" $odir/train.py; $CMD;


##############
## BB-FULL ###
##############

# odir=../outputs/iter_13-full; if [ ! -d ${odir} ]; then mkdir ${odir}; fi; python train.py --train ../data/pairs/lynntao_pdbseqs_TRAIN-SET_cmap-filtered.tsv --test ../data/pairs/esm_test.tsv --cmap ../../D-SCRIPT/data/embeddings/cmap-latest.h5 --cmap_lang ../data/emb/lynnemb/new_cmap_embed --device 6 --output_prefix ${odir}/op_ --no_bins 25 --no_epoch 25

# iter=75;mtype=-1;dev=0;odir=../outputs/iter_${iter}-bb-full; if [ ! -d ${odir} ]; then mkdir ${odir}; cp train.py model.py $odir/; fi; CMD="python train.py --train ../data/pairs/lynntao_pdbseqs_TRAIN-SET_cmap-filtered.tsv --test ../data/pairs/esm_test.tsv --cmap ../../D-SCRIPT/data/embeddings/cmap-latest.h5 --cmap_lang ../data/emb/lynnemb/new_cmap_embed  --device ${dev} --output_prefix ${odir}/op_ --input_dim 6165 --no_bins 25 --lrate 2e-4 --no_epoch 100 --activation sigmoid --model_type $mtype --cross_block 5 --conv_channels 45 --conv_kernels 5  --iter_dir ${odir}"; sed -Ei "1 i # Trained with command: $CMD" $odir/train.py; $CMD;

#ESM
#mtype=-1;dev=0;odir=../outputs/iter_19-esm; if [ ! -d ${odir} ]; then mkdir ${odir}; cp train.py model.py $odir/; fi; CMD="python train.py --train ../data/pairs/lynntao_pdbseqs_TRAIN-SET_cmap-filtered-lt1000.tsv --test ../data/pairs/esm_test.tsv --cmap ../../D-SCRIPT/data/embeddings/cmap-latest.h5 --cmap_lang ../data/emb/cmap_lang_esm.h5 --device ${dev} --output_prefix ${odir}/op_ --input_dim 1280 --no_bins 25 --lrate 1e-3 --no_epoch 50 --activation sigmoid --model_type $mtype"; sed -Ei "1 i # Trained with command: $CMD" $odir/train.py; $CMD;

#ESM-DEBUG
#mtype=-1;dev=0;odir=../outputs/iter_27-esm; if [ ! -d ${odir} ]; then mkdir ${odir}; cp train.py model.py $odir/; fi; CMD="python train.py --train ../data/pairs/human_cm_dtrain.tsv --test ../data/pairs/esm_test.tsv --cmap ../data/emb/cmap_d_emb.h5 --cmap_lang ../data/emb/cmap_lang_esm.h5 --device ${dev} --output_prefix ${odir}/op_ --input_dim 1280 --no_bins 25 --lrate 1e-3 --no_epoch 50 --activation sigmoid --model_type $mtype"; sed -Ei "1 i # Trained with command: $CMD" $odir/train.py; $CMD;

def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help = "Train File")
    parser.add_argument("--test", help = "Test File")
    parser.add_argument("--cmap", help = "Cmap outputs")
    parser.add_argument("--cmap_lang", help = "Bepler and Berger embedding")
    parser.add_argument("--input_dim", default = 6165, type = int)
    parser.add_argument("--proj_dim", default = 100, type = int)
    parser.add_argument("--no_heads", default = 5, type = int)
    parser.add_argument("--no_bins", default = 64, type = int)
    parser.add_argument("--lrate", default = 1e-5, type = float)
    parser.add_argument("--no_epoch", default = 10, type = int)
    parser.add_argument("--device", default = -1, type = int)
    parser.add_argument("--cross_block", default = 2, type = int)
    parser.add_argument("--test_image", default = 50, type = int)
    parser.add_argument("--train_image", default = 50, type = int)
    parser.add_argument("--output_prefix")
    parser.add_argument("--checkpoint", default = None)
    parser.add_argument("--activation", default = "sigmoid")
    parser.add_argument("--model_type", type = int, default = -1)
    parser.add_argument("--bins_weighting", type = int, default = -1)
    parser.add_argument("--bins_window", type = int, default = 3)
    parser.add_argument("--conv_channels", default=None)
    parser.add_argument("--conv_kernels", default=None)
    parser.add_argument("--iter_dir", required = True)
    parser.add_argument("--n_transformer_block", default = 2, type = int)
    return parser.parse_args()

def bins_weighting(option, no_bins):
    if option == 0:
        return torch.ones(no_bins, dtype = torch.float32)
    if option == 1:
        return torch.tensor([100] * 5 + [10] * 5 + [1] * (no_bins - 15) + [0.125] * 5, dtype = torch.float32)
    if option == 2:
        return torch.tensor([500] * 5 + [10] * 5 + [1] * (no_bins - 15) + [0.125] * 5, dtype = torch.float32)
    if option >= 3 or option <= -1:
        return torch.tensor([500] * 5 + [10] * 5 + [1] * (no_bins - 15) + [0.125] * 5, dtype = torch.float32)
    
def main(args):
    if args.device < 0:
        dev = torch.device("cpu")
    else:
        dev = torch.device(args.device)
        
    model_opts = {0: StructCmap, 1: StructCmapCATT, 2: WindowedStructCmapCATT, 3: WindowedStackedStructCmapCATT, 
                  4: WindowedStackedStructCmapCATT2, 5: WindowedStackedStructCmapCATT3, -1: WindowedStackedStructCmapCATT3}
    modtype = model_opts[args.model_type]
    
    if args.checkpoint is None:
        conv_kernels  = []
        conv_channels = []
        
        if args.conv_channels is not None:
            conv_channels = [int(c) for c in args.conv_channels.split(",")]
        if args.conv_kernels is not None:
            conv_kernels = [int(c) for c in args.conv_kernels.split(",")]
        
        assert len(conv_channels) == len(conv_kernels)
        
        mod = modtype(args.input_dim,
                    project_dim = args.proj_dim, 
                    n_head_within = args.no_heads,
                    n_bins = args.no_bins,
                    activation = args.activation,
                    n_crossblock = args.cross_block,
                    w_size = args.bins_window,
                    conv_channels = conv_channels, 
                    n_transformer = args.n_transformer_block,
                    kernels = conv_kernels).to(dev)
        start_epoch = 0
    else:
        epcmp = re.compile(r'.*_([0-9]+).sav$')
        start_epoch = int(epcmp.match(args.checkpoint).group(1)) + 1
        mod = torch.load(args.checkpoint, map_location = dev)
        
    image_fld = f"{args.output_prefix}images/"
    train_fld = f"{args.output_prefix}train_images/"
    metrics_fld = f"{args.output_prefix}metrics/"
    
    if not os.path.isdir(image_fld):
        print(f"[+] Creating folder {image_fld}")
        os.mkdir(image_fld)
        
    if not os.path.isdir(train_fld):
        print(f"[+] Creating folder {train_fld}")
        os.mkdir(train_fld)
        
    if not os.path.isdir(metrics_fld):
        print(f"[+] Creating folder {metrics_fld}")
        os.mkdir(metrics_fld)
    
    optim = torch.optim.Adam(mod.parameters(), lr = args.lrate) # weight decay on optimizer 1e-2
    bins_weight = bins_weighting(args.bins_weighting, args.no_bins).to(dev)
    #bins_weight = torch.tensor([50] * 5 + [10] * 5 + [1] * (args.no_bins - 15) + [0.125] * 5, dtype = torch.float32).to(dev)
    
    # TODO: smoothing window on the prob distribution
    lossf = nn.CrossEntropyLoss(weight = bins_weight)
    
    file = open(f"{args.output_prefix}results.log", "a")
    
    dtrain = PairData(args.cmap, args.cmap_lang, args.train, no_bins = args.no_bins)
    dtest = PairData(args.cmap, args.cmap_lang, args.test, no_bins = args.no_bins)
    trainloader = DataLoader(dtrain, shuffle = True, batch_size= 1)
    testloader = DataLoader(dtest, shuffle = False, batch_size = 1)
    
    no_train_batches = len(trainloader)
    no_test_batches = len(testloader)
    np.random.seed(137)
    test_indices = set(np.random.choice(no_test_batches,
                                         size = min(args.test_image, no_test_batches // 10),
                                        replace = False))
    train_indices = set(np.random.choice(no_train_batches,
                                         size = min(args.train_image, no_train_batches // 10),
                                        replace = False))
    
    tlossl = []
    loc = torch.linspace(0, 25, args.no_bins).unsqueeze(1).unsqueeze(1)
    
    wandb.watch(mod, log_freq = 100)
    
    for e in range(start_epoch, args.no_epoch):
        tloss = 0
        mod.train()
        fnat = 0
        fnnat = 0
        for i, data in enumerate(trainloader):
            score, Xp, Xq, cm = data
            optim.zero_grad()
            Xp = Xp.to(dev)
            Xq = Xq.to(dev)
            cm = cm.to(dev)
            score = score.to(dev)
            
            cpre, cppi = mod(Xp, Xq) # cpred = batch x nbins x nseq1 x nseq2
            cpred = cpre.view(1, args.no_bins, -1).contiguous() # batch x nbins x(nseq1 . nseq2)
            cpred = torch.transpose(cpred, 2, 1).squeeze() # (nseq1 . nseq2) x nbins
            loss  = lossf(cpred, cm.view(-1)) # (nseq1 . nseq2)
            tlossl.append(loss.item())
            loss.backward()
            optim.step()
            tloss += loss.item()
            
            
            if args.device > -1:
                Xp = Xp.cpu()
                Xq = Xq.cpu()
                cm = cm.cpu()
            
            perc_complete = 100 * i / no_train_batches
            if i % 100 == 0:
                file.write(f"[{e+1}/{args.no_epoch}] Training {perc_complete}% : Loss = {loss.item()}\n")
                file.flush()
                wandb.log({"train/loss" : loss})
            
            with torch.no_grad():
                cm = cm.squeeze().numpy()
                cpre = F.softmax(cpre.squeeze().cpu(), dim = 0)
                cmout  = torch.sum(cpre * loc, dim = 0).squeeze().numpy()
                cmoutbin = cmout < 10
                ctrue = cm / args.no_bins * 26 < 10
                curr_f_nat  = calc_f_nat(ctrue, cmoutbin)
                curr_f_nonnat = calc_f_nonnat(ctrue, cmoutbin)
                
                fnat += curr_f_nat
                fnnat += curr_f_nonnat
                
                ## Create images here
                if i in train_indices:                    
                    fig, ax = plt.subplots(1, 4)
                    ax[0].imshow(ctrue)
                    ax[1].imshow(cmoutbin)
                    ax[2].imshow(25 - (cm / args.no_bins * 26))
                    ax[3].imshow(25 - cmout)
                    ax[1].set_title(f"Fnat : {curr_f_nat:.3f}")
                    ax[2].set_title(f"F-nonnat : {curr_f_nonnat:.3f}")
                    fig.savefig(f"{train_fld}_img_{i}_iter{e}.png")
                    plt.close()
        tloss /= (i+1)
        fnat /= (i+1)
        fnnat /= (i+1)
        file.write(f"[+] Finished Training Epoch {e + 1}: Training CMAP loss: {tloss:.3f}, Fnat: {fnat:.5f}, F-nnat: {fnnat:.5f}\n")
        file.flush()
        wandb.log({"train/fnat" : fnat, "train/fnnat" : fnnat, "epoch": e})
        torch.save(mod, f"{args.output_prefix}model_{e}.sav")
        mod.eval()
        teloss = 0
        test_fnat = []
        test_fnnat = []
        with torch.no_grad():
            for i, data in enumerate(testloader):
                sc, Xp, Xq, cm = data
                Xp = Xp.to(dev)
                Xq = Xq.to(dev)
                cm = cm.to(dev)

                cm = cm.squeeze()

                cpred, p = mod(Xp, Xq)
                cmpred = cpred.view(1, args.no_bins, -1).contiguous()
                cmpred = torch.transpose(cmpred, 2, 1).squeeze()
                loss  = lossf(cmpred, cm.view(-1))

                teloss += loss.item()

                cpred = F.softmax(cpred.squeeze().cpu(), dim = 0)
                cmout  = torch.sum(cpred * loc, dim = 0).squeeze().numpy()
                
                if args.device > -1:
                    Xp = Xp.cpu()
                    Xq = Xq.cpu()
                    cm = cm.cpu()
                
                cm = cm.numpy()
                
                ## Metrics to be added
                ctrue = cm / args.no_bins * 26 < 10
                cpre  = cmout < 10
                curr_f_nat = calc_f_nat(ctrue, cpre)
                curr_f_nonnat = calc_f_nonnat(ctrue, cpre)
                
                test_fnat.append(curr_f_nat)
                test_fnnat.append(curr_f_nonnat)
                
                if i in test_indices:
                    fig, ax = plt.subplots(1, 4)
                    ax[0].imshow(cm / args.no_bins * 26 < 10)
                    ax[1].imshow(cmout < 10)
                    ax[2].imshow(25 - (cm / args.no_bins * 26))
                    ax[3].imshow(25 - cmout)
                    ax[1].set_title(f"Fnat : {curr_f_nat:.3f}")
                    ax[2].set_title(f"F-nonnat : {curr_f_nonnat:.3f}")
                    fig.savefig(f"{image_fld}_img_{i}_iter{e}.png")
                    plt.close()
        teloss /= (i+1)
        tfnat = np.average(test_fnat)
        tfnnat = np.average(test_fnnat)
        file.write(f"[+] Finished Testing Epoch {e + 1}:Testing CMAP loss: {teloss:.5f}, Fnat: {tfnat:.5f}, Fnnat : {tfnnat:.5f}\n")
        file.flush()
        wandb.log({"test/fnat" : tfnat, "test/fnnat" : tfnnat, "epoch" : e})
    plot_losses(tlossl, f"{metrics_fld}/losses.png")

from pathlib import Path

if __name__ == "__main__":
    args = getargs()
    oc = OmegaConf.create(vars(args))
    
    with open(f"{oc.iter_dir}/cfg.yml", "w+") as ymlf:
        ymlf.write(OmegaConf.to_yaml(oc))
        
    wandb.init(project="D-SCRIPT 3D", entity="bergerlab-mit", name = Path(oc.iter_dir).name, config = dict(oc))
    main(args)
        
    