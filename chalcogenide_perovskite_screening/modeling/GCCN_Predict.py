"""GCNN-based synthesizability prediction for crystal structures.

This module provides utilities for predicting the crystal-likeness (synthesizability)
of perovskite structures using a Graph Convolutional Neural Network (GCNN). It includes
data preparation, model loading, and inference pipelines for evaluating experimental
plausibility of computationally generated crystal structures.

Key Functions:
    - create_id_prop: Prepare crystal composition data for GCNN inference
    - run_prediction: Execute GCNN predictions on crystal structures
    - use_model: Inference function for individual model evaluation

For more details on the GCNN algorithm and methodology, see the module documentation
in chalcogenide_perovskite_screening/modeling/gcnn/README.md
"""

from __future__ import annotations

import os
import json
from time import time
from turtle import pd
from sklearn import metrics
import sys
import numpy as np
from pathlib import Path


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

from chalcogenide_perovskite_screening.modeling.gcnn.model import GCNN
from chalcogenide_perovskite_screening.modeling.gcnn.data import Parallel_Collate_Pool, get_loader, CIFData
from chalcogenide_perovskite_screening.plots import normalize_abx3

from chalcogenide_perovskite_screening.config import CRYSTALLM_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR

from glob import glob
import csv

def main():
    """Entry point for GCNN synthesizability prediction pipeline.
    
    Executes the full prediction workflow including data preparation,
    model loading, and output generation.
    """
    run_prediction()


def create_id_prop(input_data_folder: Path = CRYSTALLM_DATA_DIR / "cif_files",
                   sisso_csv: Path = PROCESSED_DATA_DIR / "results_SISSO_with_bandgap.csv",
                   crystal_csv: Path = PROCESSED_DATA_DIR / "results_CrystaLLM_with_HHI.csv",
                   exp_csv: Path = PROCESSED_DATA_DIR / "chpvk_dataset.csv",
                   output_id_prop: Path = CRYSTALLM_DATA_DIR / "cif_files/id_prop.csv") -> None:
    """Prepare composition ID and label data for GCNN inference.
    
    Merges CrystaLLM-generated structures with SISSO results and experimental data
    to create a labeled dataset for GCNN prediction. Assigns labels based on whether
    structures have experimental support in the chalcogenide perovskite dataset.
    
    Args:
        input_data_folder: Path to CIF files directory.
        sisso_csv: Path to SISSO tolerance factor results.
        crystal_csv: Path to CrystaLLM results with HHI scores.
        exp_csv: Path to experimental chalcogenide perovskite dataset.
        output_id_prop: Output path for composition-label CSV file.
    
    Returns:
        None. Writes id_prop.csv to output_id_prop path.
    """

    import pandas as pd
    from pymatgen.core.composition import Composition
    
    df_sisso = pd.read_csv(sisso_csv)
    df_crystal = pd.read_csv(crystal_csv)

    df_sisso.rename(columns={"formula": "material"}, inplace=True)

    df_sisso["norm_formula"] = df_sisso["material"].apply(normalize_abx3)
    df_crystal["norm_formula"] = df_crystal["material"].apply(normalize_abx3)

    df_crystal_sisso = df_sisso[df_sisso['norm_formula'].isin(df_crystal['norm_formula'])]

    df_crystal_sisso['Composition'] = df_crystal_sisso['material'].apply(Composition)

    df_exp = pd.read_csv(exp_csv)

    df_exp = df_exp[df_exp['exp_label'] == 1]
    df_exp['Composition'] = df_exp['material'].apply(Composition)
    df_final = df_crystal_sisso[df_crystal_sisso['Composition'].isin(df_exp['Composition'])]

    new_df = pd.DataFrame(columns=['Composition', 'label'])
    for idx, row in df_crystal_sisso.iterrows():
        comp = row['Composition'].formula.replace("1", "").replace(" ", "") + "_1"
        label = 0
        if comp in [x.formula for x in df_final['Composition']]:
            label = 1
        df = pd.DataFrame({'Composition': [comp], 'label': [label]})
        new_df = pd.concat([new_df, df], ignore_index=True)
    
    new_df.to_csv(output_id_prop, index=False, header=False)


def run_prediction(
                   input_data_folder: Path = CRYSTALLM_DATA_DIR / "cif_files",
                   folder_weights: Path = INTERIM_DATA_DIR / "weights",
                   output_prediction_json: Path = PROCESSED_DATA_DIR / "Perov_All.json.csv",
                   output_prediction_csv: Path = PROCESSED_DATA_DIR / "prediction.csv") -> None:
    """Execute GCNN-based synthesizability predictions on crystal structures.
    
    Loads pre-trained GCNN model weights, processes crystal structures from CIF files,
    and generates crystal-likeness (CL) scores with uncertainty estimates. Scores are
    computed as an ensemble average across multiple trained models for robustness.
    
    Args:
        input_data_folder: Path containing CIF files and id_prop.csv.
        folder_weights: Path containing pre-trained GCNN model weights.
        output_prediction_json: Intermediate output path for prediction JSON.
        output_prediction_csv: Final output CSV with CL scores and uncertainties.
    
    Returns:
        None. Writes prediction results to output_prediction_csv in format:
            [id, CL score, CL score std]
    
    Notes:
        - Automatically creates id_prop.csv if not present.
        - Uses GPU if available (multi-GPU supported).
        - Ensemble predictions reduce variance in synthesizability estimates.
    """
    
    data_path= input_data_folder.as_posix()

    #check if id_prop.csv exists, if not create it
    id_prop_path = input_data_folder / "id_prop.csv"
    if not id_prop_path.exists():
        create_id_prop()

    
    # Best Hyperparameters
    atom_fea_len = 64
    n_conv = 1
    lr_decay_rate = 0.99
    
    #var. for dataset loader
    batch_size = 512
    
    #var for training
    cuda = True
    
    #setup
    print('loading data...',end=''); t = time()
    data = CIFData(data_path,cache_path=data_path)
    print('completed', time()-t,'sec')
    collate_fn = Parallel_Collate_Pool(torch.cuda.device_count(),data.orig_atom_fea_len,data.nbr_fea_len)
    
    loader = get_loader(data,collate_fn,batch_size,[list(range(len(data)))],0,True)[0]
    
    #build model
    model = GCNN(data.orig_atom_fea_len,data.nbr_fea_len,atom_fea_len,n_conv)
    if cuda:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = nn.DataParallel(model)
        model.cuda()
    
    outputs = []


    for i,p in enumerate(sorted(glob(folder_weights.as_posix()+'/*_*'))):
        print('Loading model',p)
        model.load_state_dict(torch.load(p))
        output,target,mpids = use_model(loader,model,i)
        outputs.append(output)
    std = np.std(outputs,axis=0).tolist()
    #json.dump(outputs,open('predict/%s_each_score.json'%(data_path[3:].replace('/','_')),'w'))
    outputs = np.mean(outputs,axis=0).tolist()

    json.dump([mpids,outputs,target,std],open(output_prediction_json.as_posix(),'w'))

    mpids,outputs,target,std = json.load(open(output_prediction_json.as_posix(),'r'))
    with open(output_prediction_csv, 'w',newline='') as outfile:
        writer = csv.writer(outfile)
        # header
        # number, abc, label, clscore, clstd, sources
        writer.writerow(['id','CL score','CL score std'])
        for cifid,cl,clstd in zip(mpids,outputs,std):
            writer.writerow([cifid,cl,clstd])
        
    
def use_model(data_loader: object, model: nn.Module, epoch: int) -> tuple[list, list, list]:
    """Perform inference using a trained GCNN model on crystal data.
    
    Evaluates the model on all structures in the data loader in batch mode.
    Tracks timing statistics and returns crystal-likeness predictions for each structure.
    
    Args:
        data_loader: DataLoader providing batches of crystal structures and target labels.
        model: Trained GCNN model in evaluation mode.
        epoch: Epoch identifier for logging purposes.
    
    Returns:
        tuple: Contains:
            - outputs (list): Predicted crystal-likeness scores for each structure.
            - targets (list): Target labels (0/1 for unlabeled/labeled structures).
            - mpids (list): Structure identifiers, sorted in same order as outputs.
    
    Notes:
        - Model is set to eval mode (no gradient computation).
        - GPU inference used if model parameters are on CUDA device.
        - Outputs are sorted by structure ID for consistent ordering.
    """
    
    batch_time = AverageMeter()
    
    model.eval()
        
    t0 = time()
    outputs = []
    targets = []
    mpids = []
    Bs = []
    for i, (inputs,target,mpid,_) in enumerate(data_loader):
        targets += target.cpu().tolist()
        mpids += mpid
        # move input to cuda
        if next(model.parameters()).is_cuda:
            for j in range(len(inputs)): inputs[j] = inputs[j].to(device='cuda')
            target = target.to(device='cuda')
            
        #compute output
        with torch.no_grad():
            output,Weights = model(*inputs)
        outputs += output.cpu().tolist()
 
        #measure elapsed time
        batch_time.update(time() - t0)
        t0 = time()
        
        s = 'Pred '
        
        print(s+': [{0}][{1}/{2}]\t'
          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
          epoch, i, len(data_loader), batch_time=batch_time))
    
    print(s+' end: [{0}]\t'
      'Time {batch_time.sum:.3f}'.format(epoch, batch_time=batch_time))
    
    idx = np.argsort(mpids)
    outputs = [outputs[i] for i in idx]
    targets = [targets[i] for i in idx]
    mpids = [mpids[i] for i in idx]
    #Bs = [Bs[i] for i in idx]
    return outputs,targets,mpids
    #return outputs,targets,mpids,Bs
class AverageMeter(object):
    """Track average and current value of metrics.
    
    Utility class for monitoring prediction timing statistics during model inference.
    Maintains running sum, count, and average of observed values.
    """
    def __init__(self):
        """Initialize meter with zeroed values."""
        self.reset()

    def reset(self):
        """Reset all accumulated values to zero."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """Update meter with new value(s).
        
        Args:
            val: New value to add.
            n: Number of occurrences of this value (default: 1).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    # run neural network
    main()
    # compile result
    mpids,outputs,target,std = json.load(open('predict/Perov_All.json','r'))
    with open('prediction.csv', 'w',newline='') as outfile:
        writer = csv.writer(outfile)
        # header
        # number, abc, label, clscore, clstd, sources
        writer.writerow(['id','CL score','CL score std'])
        for cifid,cl,clstd in zip(mpids,outputs,std):
            writer.writerow([cifid,cl,clstd])
        
