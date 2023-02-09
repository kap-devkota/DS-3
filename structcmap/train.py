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

from model import StructCmap, StructCmapCATT, WindowedStructCmapCATT, WindowedStackedStructCmapCATT, WindowedStackedStructCmapCATT2, WindowedStackedStructCmapCATT3, WindowedStackedStructCmapCATT4
from dataset import PairData
from metrics import calc_f_nat, calc_f_nonnat, calc_top_k_precision, calc_top_Ldiv_precision, FNat, FNonNat, TopKPrecision, TopLPrecision
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

#iter=95;mtype=-1;dev=1;odir=../outputs/iter_${iter}-bb-debug-tformer; if [ ! -d ${odir} ]; then mkdir ${odir}; cp train.py model.py $odir/; fi; CMD="python train.py --train ../data/pairs/human_tr.tsv --test ../data/pairs/esm_test.tsv --cmap ../../D-SCRIPT/data/embeddings/cmap-latest.h5 --cmap_lang ../data/emb/lynnemb/new_cmap_embed  --device ${dev} --output_prefix ${odir}/op_ --input_dim 1280 --no_bins 25 --lrate 1e-3 --no_epoch 50 --activation sigmoid --model_type $mtype --cross_block 2 --conv_channels 45 --conv_kernels 5 --iter_dir ${odir} --n_transformer_block 5"; sed -Ei "1 i # Trained with command: $CMD" $odir/train.py; $CMD;


##############
## BB-FULL ###
##############

# odir=../outputs/iter_13-full; if [ ! -d ${odir} ]; then mkdir ${odir}; fi; python train.py --train ../data/pairs/lynntao_pdbseqs_TRAIN-SET_cmap-filtered.tsv --test ../data/pairs/esm_test.tsv --cmap ../../D-SCRIPT/data/embeddings/cmap-latest.h5 --cmap_lang ../data/emb/lynnemb/new_cmap_embed --device 6 --output_prefix ${odir}/op_ --no_bins 25 --no_epoch 25

# iter=75;mtype=-1;dev=0;odir=../outputs/iter_${iter}-bb-full; if [ ! -d ${odir} ]; then mkdir ${odir}; cp train.py model.py $odir/; fi; CMD="python train.py --train ../data/pairs/lynntao_pdbseqs_TRAIN-SET_cmap-filtered.tsv --test ../data/pairs/esm_test.tsv --cmap ../../D-SCRIPT/data/embeddings/cmap-latest.h5 --cmap_lang ../data/emb/lynnemb/new_cmap_embed  --device ${dev} --output_prefix ${odir}/op_ --input_dim 6165 --no_bins 25 --lrate 2e-4 --no_epoch 100 --activation sigmoid --model_type $mtype --cross_block 5 --conv_channels 45 --conv_kernels 5  --iter_dir ${odir}"; sed -Ei "1 i # Trained with command: $CMD" $odir/train.py; $CMD;

#ESM
#mtype=-1;dev=0;odir=../outputs/iter_19-esm; if [ ! -d ${odir} ]; then mkdir ${odir}; cp train.py model.py $odir/; fi; CMD="python train.py --train ../data/pairs/lynntao_pdbseqs_TRAIN-SET_cmap-filtered-lt1000.tsv --test ../data/pairs/esm_test.tsv --cmap ../../D-SCRIPT/data/embeddings/cmap-latest.h5 --cmap_lang ../data/emb/cmap_lang_esm.h5 --device ${dev} --output_prefix ${odir}/op_ --input_dim 1280 --no_bins 25 --lrate 1e-3 --no_epoch 50 --activation sigmoid --model_type $mtype"; sed -Ei "1 i # Trained with command: $CMD" $odir/train.py; $CMD;

#ESM-DEBUG
#mtype=-1;dev=0;odir=../outputs/iter_27-esm; if [ ! -d ${odir} ]; then mkdir ${odir}; cp train.py model.py $odir/; fi; CMD="python train.py --train ../data/pairs/human_cm_dtrain.tsv --test ../data/pairs/esm_test.tsv --cmap ../data/emb/cmap_d_emb.h5 --cmap_lang ../data/emb/cmap_lang_esm.h5 --device ${dev} --output_prefix ${odir}/op_ --input_dim 1280 --no_bins 25 --lrate 1e-3 --no_epoch 50 --activation sigmoid --model_type $mtype"; sed -Ei "1 i # Trained with command: $CMD" $odir/train.py; $CMD;

#iter=95;mtype=-1;dev=1;odir=../outputs/iter_${iter}-esm-mid-tformer; if [ ! -d ${odir} ]; then mkdir ${odir}; cp train.py model.py $odir/; fi; CMD="python train.py --train ../data/pairs/human_tr.tsv --test ../data/pairs/esm_test-lt1000.tsv --cmap ../../D-SCRIPT/data/embeddings/cmap-latest.h5 --cmap_lang ../data/emb/cmap_lang_esm.h5   --device ${dev} --output_prefix ${odir}/op_ --input_dim 1280 --no_bins 25 --lrate 1e-3 --no_epoch 50 --activation sigmoid --model_type $mtype --cross_block 2 --conv_channels 45 --conv_kernels 5 --iter_dir ${odir} --n_transformer_block 5"; sed -Ei "1 i # Trained with command: $CMD" $odir/train.py; $CMD;

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
    parser.add_argument("--conv_channels", default = None)
    parser.add_argument("--conv_kernels", default = None)
    parser.add_argument("--iter_dir", required = True)
    parser.add_argument("--weight_decay", default=1e-12, type = float)
    parser.add_argument("--dropout", default = 0.15, type = float)
    # parser.add_argument("--dist_thres", default = 8, type = float)
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
    
    set_random_seed(args.random_state)
    
    if args.device < 0:
        dev = torch.device("cpu")
    else:
        dev = torch.device(args.device)
        
    model_opts = {0: WindowedStackedStructCmapCATT, 
                  1: WindowedStackedStructCmapCATT2, 
                  2: WindowedStackedStructCmapCATT3, 
                  3: WindowedStackedStructCmapCATT4, 
                  -1: WindowedStackedStructCmapCATT4}
    modtype = model_opts[args.model_type]
    
    if args.checkpoint is None:
        conv_kernels  = []
        conv_channels = []
        
        if args.conv_channels is not None:
            conv_channels = [int(c) for c in args.conv_channels.split(",")]
        if args.conv_kernels is not None:
            conv_kernels = [int(c) for c in args.conv_kernels.split(",")]
        
        assert len(conv_channels) == len(conv_kernels)
        skip_connection = True if args.cross_block > 1 else False
        mod = modtype(args.input_dim,
                    project_dim = [# 256, 
                                   args.proj_dim], 
                    n_head_within = args.no_heads,
                    n_bins = args.no_bins,
                    activation = args.activation,
                    n_crossblock = args.cross_block,
                    w_size = args.bins_window,
                    drop = args.dropout,
                    conv_channels = conv_channels, 
                    n_transformer = args.n_transformer_block,
                    kernels = conv_kernels,
                    skip_connection = skip_connection).to(dev)
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
    
    optim = torch.optim.Adam(mod.parameters(), lr = args.lrate, weight_decay = args.weight_decay) # weight decay on optimizer 1e-2
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
    
    test_indices = set(np.random.choice(no_test_batches,
                                         size = min(args.test_image, no_test_batches // 10),
                                        replace = False))
    train_indices = set(np.random.choice(no_train_batches,
                                         size = min(args.train_image, no_train_batches // 10),
                                        replace = False))
    
    tlossl = []
    loc = torch.linspace(0, 25, args.no_bins).unsqueeze(1).unsqueeze(1)
    
    if not args.DEBUG: wandb.watch(mod, log_freq = args.wandb_freq)
    
    # Metrics
    train_metrics = {
        "train/fnat": FNat(thresh = args.angstrom_threshold),
        "train/fnnat": FNonNat(thresh = args.angstrom_threshold),
        "train/top10precision": TopKPrecision(thresh = args.angstrom_threshold, k = 10),
        "train/top50precision": TopKPrecision(thresh = args.angstrom_threshold, k = 50),
        "train/topL10precision": TopLPrecision(thresh = args.angstrom_threshold, Ldiv = 10),  
    }
    
    test_metrics = {
        "test/fnat": FNat(thresh = args.angstrom_threshold),
        "test/fnnat": FNonNat(thresh = args.angstrom_threshold),
        "test/top10precision": TopKPrecision(thresh = args.angstrom_threshold, k = 10),
        "test/top50precision": TopKPrecision(thresh = args.angstrom_threshold, k = 50),
        "test/topL10precision": TopLPrecision(thresh = args.angstrom_threshold, Ldiv = 10),
    }
    
    logg.info("Beginning training")
    for e in range(start_epoch, args.no_epoch):
        mod.train()
        
        tloss = 0
        
        for i, data in enumerate(trainloader):
            score, Xp, Xq, cm = data
            optim.zero_grad()
            Xp = Xp.to(dev)
            Xq = Xq.to(dev)
            cm = cm.to(dev)
            score = score.to(dev)
            
            r_out = mod(Xp, Xq)
            if len(r_out) == 2:
                cpre, cppi = r_out # cpred = batch x nbins x nseq1 x nseq2
            else:
                cpre = r_out
                
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
                cm = cm.squeeze()
                cpre = F.softmax(cpre.squeeze().cpu(), dim = 0)
                cmout  = torch.sum(cpre * loc, dim = 0).squeeze()
                ctrue = (cm / args.no_bins * 26)
                for met in train_metrics.values():
                    met.update(ctrue, cmout)
                
                ## Create images here
                if i in train_indices:
                    compare_cmaps(ctrue, cmout,
                                  thresh = args.angstrom_threshold,
                                  suptitle = f"Fnat : {calc_f_nat(ctrue, cmout):.3f}, Fnonnat : {calc_f_nonnat(ctrue, cmout):.3f}", 
                                  savefig_path = f"{train_fld}_img_{i}_iter{e}.png"
                                 )
                    
        tloss /= (i+1)
        
        train_metric_results = {k: met.compute() for k, met in train_metrics.items()}
        metstring = ", ".join([f"{k.split('/')[-1]}: {v:.5f}" for k,v in train_metric_results.items()])
        logg.info(f"Finished Training Epoch {e + 1}: cmap loss: {tloss:.5f} {metstring}")
        if not args.DEBUG: 
            train_metric_results["epoch"] = e
            wandb.log(train_metric_results)
            torch.save(mod, f"{args.iter_dir}/model_{e}.sav")
        
        for met in train_metrics.values():
            met.reset()
            
        mod.eval()
        teloss = 0
        with torch.no_grad():
            for i, data in enumerate(testloader):
                sc, Xp, Xq, cm = data
                Xp = Xp.to(dev)
                Xq = Xq.to(dev)
                cm = cm.to(dev)

                cm = cm.squeeze()

                cpred = mod(Xp, Xq)
                
                if len(cpred) == 2:
                    cpred, p = cpred
                
                cmpred = cpred.view(1, args.no_bins, -1).contiguous()
                cmpred = torch.transpose(cmpred, 2, 1).squeeze()
                loss   = lossf(cmpred, cm.view(-1))

                teloss += loss.item()

                cpred = F.softmax(cpred.squeeze().cpu(), dim = 0)
    
                if args.device > -1:
                    Xp = Xp.cpu()
                    Xq = Xq.cpu()
                    cm = cm.cpu()
                
                ## Metrics to be added
                cmout  = torch.sum(cpred * loc, dim = 0).squeeze()
                ctrue = (cm / args.no_bins * 26)
                
                for met in test_metrics.values():
                    met.update(ctrue, cmout)
                
                if i in test_indices:
                    compare_cmaps(ctrue, cmout,
                                  thresh = args.angstrom_threshold,
                                  suptitle = f"Fnat : {calc_f_nat(ctrue, cmout):.3f}, Fnonnat : {calc_f_nonnat(ctrue, cmout):.3f}",
                                  savefig_path = f"{image_fld}_img_{i}_iter{e}.png"
                                 )

        teloss /= (i+1)
        
        test_metric_results = {k: met.compute() for k, met in test_metrics.items()}
        metstring = ", ".join([f"{k.split('/')[-1]}: {v:.5f}" for k,v in test_metric_results.items()])
        logg.info(f"Finished Testing Epoch {e + 1}: cmap loss: {teloss:.5f} {metstring}")
        if not args.DEBUG:
            test_metric_results["epoch"] = e
            wandb.log(test_metric_results)

        for met in test_metrics.values():
            met.reset()
        
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
        
    