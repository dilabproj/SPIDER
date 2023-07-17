import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn

from jdlib.experiment import ExperimentConfig
from jdlib.experiment import run_experiment, run_evaluation

from jdlib.datasets import PTBXLDataset, PTBXLDatasetSubset
from jdlib.datasets import ChapmanDataset, ChapmanDatasetSubset
from jdlib.datasets import CPSCDataset, CPSCDatasetSubset

from jdlib.encoders import TSEncoder
from jdlib.models import SlotFormer, SlotFormerPos, SlotFormerPos2
from jdlib.models import GlobalAverPooling1d, SingleTransEncoder  # global branches
from jdlib.models import TemporalSegmentation, SlotEncoder, SlotFormerPos2  # local branches
from jdlib.losses import HungarianLoss, HungarianLoss2, HierarchicalLoss, InfoNCELoss
from jdlib.transformation import preprocess_lead, RandomSelectAugmentation
from jdlib.transformation import RandomCrop, MinMaxScalar, ZScoreScalar


config = ExperimentConfig()

# random seed
config.random_seed = 666

# wandb settings
config.wandb_project = "research-test2"

# dataset settings
config.db_dir = "/mnt/data2/nfs/chunti/ECG Datasets/ptb-xl"  # ptb-xl / chapman / cpsc2018
if config.db_dir.split("/")[-1] == "chapman":
    config.dataset = ChapmanDataset
    config.datasetSubset = ChapmanDatasetSubset
elif config.db_dir.split("/")[-1] == "ptb-xl":
    config.dataset = PTBXLDataset
    config.datasetSubset = PTBXLDatasetSubset
elif config.db_dir.split("/")[-1] == "cpsc2018":
    config.dataset = CPSCDataset
    config.datasetSubset = CPSCDatasetSubset
    
config.sampling_rate = 100
config.split_ratio = (0.7, 0.1, 0.2)
config.global_transform = None  # [MinMaxScalar(), ]
config.preprocess_lead = preprocess_lead
config.preprocess_args = {"lead_sampling_rate": config.sampling_rate}

# training settings
config.batch_size = 256
config.num_epochs = 100
config.warmup_epochs = 10
config.dataloader_num_worker = 6
config.gpu_device = "cuda:0"

# model settings
config.encoder = TSEncoder
config.encoder_args = {"input_dims": 12, 
                       "hidden_dims": 32, 
                       "output_dims": 128, 
                       "depth": 10,
                       "dilation_base": 1}

# global branch: (N, C, L) -> (N, dim)
# config.global_branch = GlobalAverPooling1d
# config.global_branch_arg = {}
config.global_branch = SingleTransEncoder
config.global_branch_args = {"dim": config.encoder_args['output_dims'], 
                             "nhead": 1, 
                             "num_layers": 1}
# config.global_branch = SlotEncoder
# config.global_branch_args = {"dim": config.encoder_args['output_dims'], 
#                              "num_slots": 10, 
#                              "num_iter": 5}

# local branch: (N, C, L) -> (N, num_repr, dim)
# config.local_branch = TemporalSegmentation
# config.local_branch_args = {"num_slots": 6}
config.local_branch = SlotEncoder
config.local_branch_args = {"dim": config.encoder_args['output_dims'], 
                            "num_slots": 6,
                            "num_iter": 5,
                            "target_time_steps": 100,
                            "k": 3,
                            "learnable_init": True}

config.model = SlotFormerPos2
config.model_args = {"augmentation": None,  # RandomSelectAugmentation(n=1),
                     "intermediate_learning_rate": 0,
                     "global_learning_rate": 1e-4,
                     "local_learning_rate": 1e-4, 
                     "intermediate_loss_func": None,
                     "global_loss_func": HierarchicalLoss(l2_norm=False),
                     "local_loss_func": HungarianLoss2(),
                     }

config.checkpoint_root = f"/mnt/data2/nfs/chunti/proposed/saved_models/{config.model.__name__}/"
if not os.path.exists(config.checkpoint_root):
    os.makedirs(config.checkpoint_root)

result_root = config.checkpoint_root.replace("saved_models", "results")
if not os.path.exists(result_root):
    os.makedirs(result_root)

# additional argument setting for convenience
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--cur_time", help="to set the run name")
parser.add_argument("-s", "--seed", help="to set the random seed", type=int)
parser.add_argument("-g", "--gpu_id", help="to set the gpu_id", type=int)
args = parser.parse_args()

if __name__ == "__main__": 
    # setting random seed
    dataset_dict = {"ptbxl": {"db_dir": "/mnt/data2/nfs/chunti/ECG Datasets/ptb-xl",
                              "dataset": PTBXLDataset,
                              "datasetSubset": PTBXLDatasetSubset},
                    "chapman": {"db_dir": "/mnt/data2/nfs/chunti/ECG Datasets/chapman",
                                "dataset": ChapmanDataset,
                                "datasetSubset": ChapmanDatasetSubset}, 
                    "cpsc": {"db_dir": "/mnt/data2/nfs/chunti/ECG Datasets/cpsc2018",
                             "dataset": CPSCDataset,
                             "datasetSubset": CPSCDatasetSubset}
                   }
    
    config.cur_time = args.cur_time if args.cur_time is not None else config.cur_time
    config.random_seed = args.seed if args.seed is not None else config.random_seed
    config.gpu_device = f"cuda:{args.gpu_id}" if args.gpu_id is not None else config.gpu_device
    
    for dataset_name in dataset_dict:
        config.db_dir = dataset_dict[dataset_name]['db_dir']
        config.dataset = dataset_dict[dataset_name]['dataset']
        config.datasetSubset = dataset_dict[dataset_name]['datasetSubset']
        
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)

        run_experiment(config)
        run_evaluation(config)