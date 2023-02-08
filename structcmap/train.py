import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import h5py
import argparse
import os
import sys
import matplotlib.pyplot as plt
import re
import logging as lg
from pathlib import Path
from omegaconf import OmegaConf
import wandb
import json
import typing as T

from model import StructCmap, StructCmapCATT, WindowedStructCmapCATT, WindowedStackedStructCmapCATT, WindowedStackedStructCmapCATT2, WindowedStackedStructCmapCATT3
from dataset import PairData
from metrics import calc_f_nat, calc_f_nonnat, calc_top_k_precision, calc_top_Ldiv_precision
from plotting import compare_cmaps, plot_fnat_histograms

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

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

logLevels = {"ERROR": lg.ERROR, "WARNING": lg.WARNING, "INFO": lg.INFO, "DEBUG": lg.DEBUG}
LOGGER_NAME = "DS-3"
    
def config_logger(
    file: T.Union[Path, None],
    fmt: str = "[%(asctime)s] %(message)s",
    level: bool = 2,
    use_stdout: bool = True,
):
    """
    Create and configure the logger
    :param file: Can be a Path or None -- if a Path, log messages will be written to the file at Path
    :type file: T.Union[Path, None]
    :param fmt: Formatting string for the log messages
    :type fmt: str
    :param level: Level of verbosity
    :type level: int
    :param use_stdout: Whether to also log messages to stdout
    :type use_stdout: bool
    :return:
    """

    module_logger = lg.getLogger(LOGGER_NAME)
    module_logger.setLevel(logLevels[level])
    formatter = lg.Formatter(fmt)

    if file is not None:
        fh = lg.FileHandler(file)
        fh.setFormatter(formatter)
        module_logger.addHandler(fh)

    if use_stdout:
        sh = lg.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        module_logger.addHandler(sh)

    lg.propagate = False

    return module_logger

def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help = "Train File (.tsv)")
    parser.add_argument("--test", help = "Test File (.tsv)")
    parser.add_argument("--cmap", help = "Cmap outputs (.h5)")
    parser.add_argument("--cmap_lang", help = "Language model embedding (.h5)")
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
    parser.add_argument("--checkpoint", default = None)
    parser.add_argument("--activation", default = "sigmoid")
    parser.add_argument("--model_type", type = int, default = -1)
    parser.add_argument("--bins_weighting", type = int, default = -1)
    parser.add_argument("--bins_window", type = int, default = 3)
    parser.add_argument("--conv_channels", default=None)
    parser.add_argument("--conv_kernels", default=None)
    parser.add_argument("--iter_dir", required = True)
    parser.add_argument("--n_transformer_block", default = 2, type = int)
    parser.add_argument("--angstrom_threshold", default = 8, type = float)
    parser.add_argument("--random_state", default = 317, type = int)
    parser.add_argument("--wandb_freq", default = 1000, type = int)
    parser.add_argument("--DEBUG", action = "store_true", help = "Runs will not be logged to Weights&Biases, models will not be saved")
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
    logg = lg.getLogger(LOGGER_NAME)
    logg.info(json.dumps(vars(args), indent=4))
    
    logg.info(f"Logging to {args.iter_dir}/results.log")
    
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
        
    image_fld = f"{args.iter_dir}/images/"
    train_fld = f"{args.iter_dir}/train_images/"
    
    if not os.path.isdir(image_fld):
        logg.info(f"Creating folder {image_fld}")
        os.mkdir(image_fld)
        
    if not os.path.isdir(train_fld):
        logg.info(f"Creating folder {train_fld}")
        os.mkdir(train_fld)
    
    optim = torch.optim.Adam(mod.parameters(), lr = args.lrate) # weight decay on optimizer 1e-2
    bins_weight = bins_weighting(args.bins_weighting, args.no_bins).to(dev)
    #bins_weight = torch.tensor([50] * 5 + [10] * 5 + [1] * (args.no_bins - 15) + [0.125] * 5, dtype = torch.float32).to(dev)
    
    # TODO: smoothing window on the prob distribution
    lossf = nn.CrossEntropyLoss(weight = bins_weight)
    
    dtrain = PairData(args.cmap, args.cmap_lang, args.train, no_bins = args.no_bins)
    dtest = PairData(args.cmap, args.cmap_lang, args.test, no_bins = args.no_bins)
    trainloader = DataLoader(dtrain, shuffle = True, batch_size= 1)
    testloader = DataLoader(dtest, shuffle = False, batch_size = 1)
    
    no_train_batches = len(trainloader)
    no_test_batches = len(testloader)
    set_random_seed(args.random_state)
    test_indices = set(np.random.choice(no_test_batches,
                                         size = min(args.test_image, no_test_batches // 10),
                                        replace = False))
    train_indices = set(np.random.choice(no_train_batches,
                                         size = min(args.train_image, no_train_batches // 10),
                                        replace = False))
    
    tlossl = []
    loc = torch.linspace(0, 25, args.no_bins).unsqueeze(1).unsqueeze(1)
    
    if not args.DEBUG: wandb.watch(mod, log_freq = args.wandb_freq)
    
    logg.info("Beginning training")
    for e in range(start_epoch, args.no_epoch):
        tloss = 0
        mod.train()
        fnat = 0
        fnnat = 0
        train_topprecision = []
        
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
                logg.info(f"[{e+1}/{args.no_epoch}] Training {perc_complete}% : Loss = {loss.item()}")
            if i % args.wandb_freq:
                if not args.DEBUG: wandb.log({"train/loss" : loss})
            
            with torch.no_grad():
                cm = cm.squeeze().numpy()
                cpre = F.softmax(cpre.squeeze().cpu(), dim = 0)
                cmout  = torch.sum(cpre * loc, dim = 0).squeeze().numpy()
                ctrue = (cm / args.no_bins * 26)
                    
                curr_f_nat  = calc_f_nat(ctrue, cmout, thresh = args.angstrom_threshold)
                curr_f_nonnat = calc_f_nonnat(ctrue, cmout, thresh = args.angstrom_threshold)
                curr_top10_prec = calc_top_k_precision(ctrue, cmout, thresh = args.angstrom_threshold, k = 10)
                curr_top50_prec = calc_top_k_precision(ctrue, cmout, thresh = args.angstrom_threshold, k = 50)
                curr_topL10_prec = calc_top_Ldiv_precision(ctrue, cmout, thresh = args.angstrom_threshold, Ldiv = 10)
                
                fnat += curr_f_nat
                fnnat += curr_f_nonnat
                train_topprecision.append([curr_top10_prec, curr_top50_prec, curr_topL10_prec])
                
                ## Create images here
                if i in train_indices:
                    compare_cmaps(ctrue, cmout,
                                  thresh = args.angstrom_threshold,
                                  suptitle = f"Fnat : {curr_f_nat:.3f}, Fnonnat : {curr_f_nonnat:.3f}", 
                                  savefig_path = f"{train_fld}_img_{i}_iter{e}.png"
                                 )
                    
        tloss /= (i+1)
        fnat /= (i+1)
        fnnat /= (i+1)
        topprec = np.average(train_topprecision, axis=0)
        
        logg.info(f"Finished Training Epoch {e + 1}: Training CMAP loss: {tloss:.3f}, Fnat: {fnat:.5f}, F-nnat: {fnnat:.5f}, Top (10, 50, L/10) precision: {topprec}")
        if not args.DEBUG: wandb.log({"train/fnat" : fnat, "train/fnnat" : fnnat, "epoch": e})
        if not args.DEBUG: torch.save(mod, f"{args.iter_dir}/model_{e}.sav")
        mod.eval()
        teloss = 0
        test_fnat = []
        test_fnnat = []
        test_topprecision = []
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
                if args.device > -1:
                    Xp = Xp.cpu()
                    Xq = Xq.cpu()
                    cm = cm.cpu()
                cm = cm.numpy()
                
                ## Metrics to be added
                cmout  = torch.sum(cpred * loc, dim = 0).squeeze().numpy()
                ctrue = (cm / args.no_bins * 26)
                curr_f_nat  = calc_f_nat(ctrue, cmout, thresh = args.angstrom_threshold)
                curr_f_nonnat = calc_f_nonnat(ctrue, cmout, thresh = args.angstrom_threshold)
                curr_top10_prec = calc_top_k_precision(ctrue, cmout, thresh = args.angstrom_threshold, k = 10)
                curr_top50_prec = calc_top_k_precision(ctrue, cmout, thresh = args.angstrom_threshold, k = 50)
                curr_topL10_prec = calc_top_Ldiv_precision(ctrue, cmout, thresh = args.angstrom_threshold, Ldiv = 10)
                
                test_fnat.append(curr_f_nat)
                test_fnnat.append(curr_f_nonnat)
                test_topprecision.append([curr_top10_prec, curr_top50_prec, curr_topL10_prec])
                
                if i in test_indices:
                    compare_cmaps(ctrue, cmout,
                                  thresh = args.angstrom_threshold,
                                  suptitle = f"Fnat : {curr_f_nat:.3f}, Fnonnat : {curr_f_nonnat:.3f}", 
                                  savefig_path = f"{image_fld}_img_{i}_iter{e}.png"
                                 )

        teloss /= (i+1)
        tfnat = np.average(test_fnat)
        tfnnat = np.average(test_fnnat)
        ttopprec = np.average(test_topprecision, axis=0)
        
        # Fnat/Fnnat histograms for test set
        plot_fnat_histograms(test_fnat, test_fnnat, f"{image_fld}_iter{e}_histograms.png")
        
        logg.info(f"Finished Testing Epoch {e + 1}:Testing CMAP loss: {teloss:.5f}, Fnat: {tfnat:.5f}, Fnnat : {tfnnat:.5f}, Top (10, 50, L/10) precision: {ttopprec}")
        if not args.DEBUG: wandb.log({"test/fnat" : tfnat, "test/fnnat" : tfnnat, "epoch" : e})

if __name__ == "__main__":
    # Read Arguments
    args = getargs()
    oc = OmegaConf.create(vars(args))
    
    # Create Project Directory
    try:
        os.makedirs(Path(oc.iter_dir))
    except FileExistsError:
        raise FileExistsError(f"Results already exist at {oc.iter_dir} - would be overwritten.")
    
    # Initialize Logging
    oc.log_level = "DEBUG" if args.DEBUG else "INFO"
    config_logger(
        file = f"{oc.iter_dir}/results.log",
        level = oc.log_level,
    )
    if not args.DEBUG: wandb.init(project="D-SCRIPT 3D", entity="bergerlab-mit", name = Path(oc.iter_dir).name, config = dict(oc))
    
    # Write Config
    with open(f"{oc.iter_dir}/cfg.yml", "w+") as ymlf:
        ymlf.write(OmegaConf.to_yaml(oc))
    
    # MAIN
    main(args)
        
    