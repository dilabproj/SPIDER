import logging

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Callable, Optional, Dict, Tuple, Any, Type

import torch
import torch.nn as nn


def transform_dict(config_dict: Dict, expand: bool = True):
    """
    General function to transform any dictionary into wandb config acceptable format
    (This is mostly due to datatypes that are not able to fit into YAML format which makes wandb angry)
    The expand argument is used to expand iterables into dictionaries
    So the configs can be used when comparing results across runs
    """
    ret: Dict[str, Any] = {}
    for k, v in config_dict.items():
        if v is None or isinstance(v, (int, float, str)):
            ret[k] = v
        elif isinstance(v, (list, tuple, set)):
            # Need to check if item in iterable is YAML-friendly
            t = transform_dict(dict(enumerate(v)), expand)
            # Transform back to iterable if expand is False
            ret[k] = t if expand else [t[i] for i in range(len(v))]
        elif isinstance(v, dict):
            ret[k] = transform_dict(v, expand)
        else:
            # Transform to YAML-friendly (str) format
            # Need to handle both Classes, Callables, Object Instances
            # Custom Classes might not have great __repr__ so __name__ might be better in these cases
            try:
                vname = v.__repr__() if hasattr(v, '__repr__') else v.__class__.__name__
            except:
                vname = v.__name__ if hasattr(v, '__name__') else v.__class__.__name__
            # ret[k] = f"{v.__module__}:{vname}"
            ret[k] = f"{vname}"
    return ret


def dfac_cur_time():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def dfac_dataset_optimizer_args():
    return {
        "lr": 1e-4,
    }


def dfac_model_args():
    return {}


def dfac_lr_scheduler_args():
    return {
        'step_size': 3,
        'gamma': 0.1,
    }



@dataclass
class ExperimentConfig:  # pylint: disable=too-many-instance-attributes
    # Data Sources
    db_dir: str = "/mnt/data2/nfs/chunti/ECG Datasets/ptb-xl"
    sampling_rate: int = 100

    # GPU Device Setting
    gpu_device: str = "cuda:0"

    # Logging Related
    cur_time: str = field(default_factory=dfac_cur_time)

    # WandB setting
    wandb_repo: str = "peterchou"
    wandb_project: str = "research"

    # Set random seed. Set to None to create new Seed
    random_seed: Optional[int] = 0
    
    # Dataset Setting
    dataset = None
    datasetSubset = None
    
    # Default No Lead Preprocessing Function
    preprocess_lead: Optional[Callable] = None
    preprocess_args: Optional[Dict[str, Any]] = None

    # Transform Function
    global_transform: Optional[Tuple[Callable, ...]] = None
    train_transform: Optional[Tuple[Callable, ...]] = None
    valid_transform: Optional[Tuple[Callable, ...]] = None
    test_transform: Optional[Tuple[Callable, ...]] = None
    
    # Default Target Attr
    # target_table: Base = ECGtoK
    target_attr: str = None
    
    # Cross Validation Split
    split_ratio = (0.7, 0.1, 0.2)
    cv_split: Optional[Tuple[int, int]] = None

    # Training Related
    batch_size: int = 64
    dataloader_num_worker: int = 0

    # Pretraining loss functions
    # slot_loss: Optional[nn.Module] = None
    # global_loss: Optional[nn.Module] = None
    
    # Encoder settings
    encoder: Optional[Type[torch.nn.Module]] = None
    encoder_args: Dict[str, Any] = field(default_factory=dict)
    
    # Global branch settings
    global_branch: Optional[Type[torch.nn.Module]] = None
    global_branch_args: Dict[str, Any] = field(default_factory=dict)
    
    # Encoder settings
    local_branch: Optional[Type[torch.nn.Module]] = None
    local_branch_args: Dict[str, Any] = field(default_factory=dict)
    
    # Default Don't Select Model
    model: Optional[Type[torch.nn.Module]] = None
    model_args: Dict[str, Any] = field(default_factory=dict)
    
    # Default model save root
    checkpoint_root: Optional[str] = None

    # Default Select Adam as Optimizer
    # optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam  # type: ignore
    # optimizer_args: Dict[str, Any] = field(default_factory=dfac_dataset_optimizer_args)

    # Default adjust learning rate
    # lr_scheduler: Optional[Type[torch.optim.lr_scheduler._LRScheduler]] = None  # pylint: disable=protected-access
    # lr_scheduler_args: Dict[str, Any] = field(default_factory=dfac_lr_scheduler_args)

    # Set number of epochs to train
    num_epochs: int = 5
    warmup_epochs: int = 0

    def target_name_map(self, ntype: str = "short") -> Dict[int, str]:
        if self.target_attr_transform is None:
            if self.target_attr in NAME_MAPPING_MAPPING:
                return NAME_MAPPING_MAPPING[self.target_attr][ntype]
            else:
                logger.warning("Target Attribute Not Transformed and target_attr %s not recognized!",
                               self.target_attr)
                return {}
        else:
            if self.target_attr_transform.__name__ in NAME_MAPPING_MAPPING:
                return NAME_MAPPING_MAPPING[self.target_attr_transform.__name__][ntype]
            else:
                logger.warning("Target Attribute Transformed and Name %s not recognized in mapping!",
                               self.target_attr_transform.__name__)
            return {}

    def to_dict(self, expand: bool = True):
        return transform_dict(asdict(self), expand)
