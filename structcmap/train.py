import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import h5py
import argparse
from model import StructCmap, StructCmapCATT, WindowedStructCmapCATT
from dataset import PairData
import os
import matplotlib.pyplot as plt
import re
from metrics import calc_f_nat, calc_f_nonnat
from plotting import plot_losses
# python train.py --train ../data/pairs/human_cm_dtrain.tsv --test ../data/pairs/human_cm_dtrain.tsv --cmap ../data/emb/cmap_d_emb.h5 --cmap_lang ../data/emb/cmap_d_lang_emb.h5 --device 3 --output_prefix ../outputs/iter_1/op_
## Full
# python train.py --train ../data/pairs/lynntao_pdbseqs_TRAIN-SET_cmap-filtered.tsv --test ../data/pairs/lynntao_pdbseqs_TEST-SET_cmap-filtered.tsv --cmap ../../D-SCRIPT/data/embeddings/cmap-latest.h5 --cmap_lang ../data/emb/lynnemb/new_cmap_embed --device 3 --output_prefix ../outputs/iter_3-train_full/op_


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
    parser.add_argument("--test_image", default = 50, type = int)
    parser.add_argument("--train_image", default = 50, type = int)
    parser.add_argument("--output_prefix")
    parser.add_argument("--checkpoint", default = None)
    parser.add_argument("--model_type", type = int, default = -1)
    return parser.parse_args()

def main(args):
    if args.device < 0:
        dev = torch.device("cpu")
    else:
        dev = torch.device(args.device)
        
    model_opts = {0: StructCmap, 1: StructCmapCATT, 2: WindowedStructCmapCATT, -1: WindowedStructCmapCATT}
    modtype = model_opts[args.model_type]
    
    if args.checkpoint is None:
        mod = modtype(args.input_dim,
                    args.proj_dim, 
                    args.no_heads,
                    args.no_bins).to(dev)
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
    
    optim = torch.optim.Adam(mod.parameters(), lr = args.lrate)
    bins_weight = torch.ones(args.no_bins, dtype = torch.float32).to(dev)
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
    test_indices = set(np.random.choice(no_test_batches,
                                         size = min(args.test_image, no_test_batches // 10),
                                        replace = False))
    
    train_indices = set(np.random.choice(no_train_batches,
                                         size = min(args.train_image, no_train_batches // 10),
                                        replace = False))
    
    tlossl = []
    loc = torch.linspace(0, 25, args.no_bins).unsqueeze(1).unsqueeze(1)
    
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
            
            with torch.no_grad():
                cm = cm.squeeze().numpy()
                cpre = F.softmax(cpre.squeeze().cpu(), dim = 0)
                cmout  = (torch.sum(cpre * loc, dim = 0) / 26 * args.no_bins).long().squeeze().numpy()
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
                cmout  = (torch.sum(cpred * loc, dim = 0) / 26 * args.no_bins).long().squeeze().numpy()
                
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
    plot_losses(tlossl, f"{metrics_fld}/losses.png")
    
if __name__ == "__main__":
    main(getargs())
        
    