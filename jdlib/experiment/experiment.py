import os
import json
import wandb
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from jdlib.datasets import PTBXLDataset, PTBXLDatasetSubset
from jdlib.tasks import run_classification


def run_experiment(config):
    # inititialize wandb
    print(f"Run name: {config.cur_time}")
    
    dataset_name = config.db_dir.split("/")[-1]
    wandb_logger = wandb.init(project=config.wandb_project,
                              name=f"{config.cur_time}_{dataset_name}",
                              config=config.to_dict())
    
    # loading dataset
    ds = config.dataset(db_dir=config.db_dir, 
                      sampling_rate=config.sampling_rate, 
                      split_ratio=config.split_ratio,
                      preprocess_lead=config.preprocess_lead,
                      preprocess_args=config.preprocess_args,
                      transform=config.global_transform,
                      target_attr=config.target_attr, 
                      timestep_first=False, 
                      random_seed=config.random_seed)
    
    train_ds = config.datasetSubset(ds, "train")
    valid_ds = config.datasetSubset(ds, "valid")
    test_ds = config.datasetSubset(ds, "test")
    
    train_loader = DataLoader(train_ds,      
                              batch_size=config.batch_size, 
                              shuffle=True, drop_last=True,
                              num_workers=config.dataloader_num_worker)

    valid_loader = DataLoader(valid_ds, 
                              batch_size=config.batch_size, 
                              shuffle=False,
                              num_workers=config.dataloader_num_worker)

    test_loader = DataLoader(test_ds, 
                             batch_size=config.batch_size, 
                             shuffle=False,
                             num_workers=config.dataloader_num_worker)
    
    # Pre-training
    model = config.model(encoder=config.encoder,
                         encoder_args=config.encoder_args,
                         global_branch=config.global_branch,
                         global_branch_args=config.global_branch_args,
                         local_branch=config.local_branch,
                         local_branch_args=config.local_branch_args,
                         epochs=config.num_epochs,
                         warmup_epochs=config.warmup_epochs,
                         device=config.gpu_device,
                         **config.model_args)
    
    saved_dir = os.path.join(config.checkpoint_root, f"{config.cur_time}_{dataset_name}")
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    saved_path = os.path.join(saved_dir, f"{config.random_seed}.pt")
    model.fit(train_loader, valid_loader, wandb_logger=wandb_logger, saved_path=saved_path)
    
    # saved_path = os.path.join(config.checkpoint_root, f"{config.cur_time}.pt")
    # model.save(saved_path)
    # print(f"The trained model is saved as {saved_path}.")
    
    with torch.cuda.device(config.gpu_device):
        torch.cuda.empty_cache()
    
    return None


def run_evaluation(config, used_feature="both", wandb_logger=True):
    ds = config.dataset(db_dir=config.db_dir, 
                      sampling_rate=config.sampling_rate, 
                      split_ratio=config.split_ratio,
                      preprocess_lead=config.preprocess_lead,  
                      preprocess_args=config.preprocess_args,
                      target_attr=config.target_attr, 
                      timestep_first=False, 
                      random_seed=config.random_seed)
    
    train_ds = config.datasetSubset(ds, "train")
    valid_ds = config.datasetSubset(ds, "valid")
    test_ds = config.datasetSubset(ds, "test")
    
    train_loader = DataLoader(train_ds,      
                              batch_size=config.batch_size, 
                              shuffle=False, 
                              num_workers=config.dataloader_num_worker)

    valid_loader = DataLoader(valid_ds, 
                              batch_size=config.batch_size, 
                              shuffle=False, 
                              num_workers=config.dataloader_num_worker)

    test_loader = DataLoader(test_ds, 
                             batch_size=config.batch_size, 
                             shuffle=False, 
                             num_workers=config.dataloader_num_worker)
    
    # Pre-training
    model = config.model(encoder=config.encoder,
                         encoder_args=config.encoder_args,
                         global_branch=config.global_branch,
                         global_branch_args=config.global_branch_args,
                         local_branch=config.local_branch,
                         local_branch_args=config.local_branch_args,
                         epochs=config.num_epochs,
                         warmup_epochs=config.warmup_epochs,
                         device=config.gpu_device,
                         **config.model_args)
    
    
    dataset_name = config.db_dir.split("/")[-1]
    saved_dir = os.path.join(config.checkpoint_root, f"{config.cur_time}_{dataset_name}")
    saved_path = os.path.join(saved_dir, f"{config.random_seed}.pt")
    model.load(saved_path)
    
#     agg_df = pd.read_csv(os.path.join(config.db_dir, "scp_statements.csv"), index_col=0)
#     all_labels = agg_df.index.values
    
#     form_labels = ['NDT', 'NST_', 'DIG', 'LNGQT', 'ABQRS', 'PVC', 'STD_', 
#                    'VCLVH', 'QWAVE', 'LOWT', 'NT_', 'PAC', 'LPR', 'INVT', 
#                    'LVOLT', 'HVOLT', 'TAB_', 'STE_','PRC(S)']

#     rhythm_labels = ['SR', 'AFIB', 'STACH', 'SARRH', 'SBRAD', 'PACE', 'SVARR', 
#                      'BIGU','AFLT', 'SVTAC', 'PSVT', 'TRIGU']
    
    
    results = run_classification(model=model, 
                                 train_loader=train_loader,
                                 test_loader=test_loader,
                                 used_feature=used_feature)
    
    with torch.cuda.device(config.gpu_device):
        torch.cuda.empty_cache()
    
    # save task results in json
    result_root = saved_dir.replace("saved_models", "results")
    if not os.path.exists(result_root):
        os.makedirs(result_root)
    file_name = os.path.join(result_root, f"{config.random_seed}.json")
    
    if not os.path.exists(file_name):
        with open(file_name, "w") as f:
            json.dump(results, f)
        
    # Labels
    aurocs, auprcs, names, label_type = [[], [], [], []]

    for k in results:
        if k in ['SR', 'NORM', 'SI', 'Normal', 'NONE']:
            continue

        names.append(k)
        aurocs.append(results[k]['auroc'])
        auprcs.append(results[k]['auprc'])

#         if k in form_labels:
#             label_type.append('form')

#         elif k in rhythm_labels:
#             label_type.append('rhythm')

#         else:
#             label_type.append('diagnostic')

    aurocs = np.array(aurocs)
    auprcs = np.array(auprcs)
#     label_type = np.array(label_type)
    
    result = {f"{dataset_name}_overall_auroc": aurocs.mean(),
#               "ptbxl_rhythm_auroc": aurocs[label_type=="rhythm"].mean(),
#               "ptbxl_form_auroc": aurocs[label_type=="form"].mean(),
#               "ptbxl_diagnostic_roc": aurocs[label_type=="diagnostic"].mean(),
              f"{dataset_name}_overall_auprc": auprcs.mean(),
#               "ptbxl_rhythm_auprc": auprcs[label_type=="rhythm"].mean(),
#               "ptbxl_form_auprc": auprcs[label_type=="form"].mean(),
#               "ptbxl_diagnostic_auprc": auprcs[label_type=="diagnostic"].mean()
             }

    if wandb_logger:
        wandb.log(result)
        wandb.finish()
    
    else:
        for k in result:
            print(f"{k}: {result[k]:.6f}")
    
    
    return results
