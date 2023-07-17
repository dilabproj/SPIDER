import os, sys, random, ast, math

from typing import Any, Tuple, List, Dict, Sequence, Optional, Callable, Union, ClassVar

import pandas as pd
import numpy as np
import wfdb
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, sampler


class BaseECGDataset(Dataset):
    def __init__(self, 
                 db_dir, 
                 sampling_rate=500,
                 target_attr=None,
                 split_ratio: Tuple[float, float, float]=(0.7, 0.1, 0.2),
                 cv_split: Optional[Tuple[int, int]]=None,
                 preprocess_lead: Optional[Callable]=None,
                 preprocess_args: Optional[Dict[str, Any]]=None,
                 timestep_first=False,
                 transform: Optional[Tuple[Callable, ...]]=None,
                 random_seed: int=666):
        super(BaseECGDataset, self).__init__(db_dir)
        
        self.db_dir = db_dir
        self.sampling_rate = sampling_rate
        assert self.sampling_rate in [100, 500]
        
        self.target_attr = target_attr
        
        self.preprocess_lead = preprocess_lead
        self.preprocess_args = preprocess_args
        self.timestep_first = timestep_first
        
        self.transform = transform
        self.split_ratio = split_ratio
        self.cv_split = cv_split
        self.random_seed = random_seed
        
        self.ecg_data = self._load_data()
        self.split_indices_dict = self._split()
        
    def _load_data(self):
        # loading ecg_data and labels
        ecg_data = pd.DataFrame()
        ecg_label = pd.DataFrame()
        
        return ecg_data, ecg_label
    
    def _split(self):
        train_id, valid_id, test_id = [[], [], []]
        
        assert set(train_id) | set(valid_id) | set(test_id) == set(range(self.__len__()))
        assert set(train_id) & set(valid_id) == set()
        assert set(train_id) & set(test_id) == set()
        assert set(valid_id) & set(test_id) == set()
        
        return {"train": train_id, "valid": valid_id, "test": test_id}
        
    def __len__(self):
        return self.ecg_data.shape[0]

    def __getitem__(self, idx):
        lead_data = self.ecg_data[]
        lead_data = self.ecg_label[]
        
        return lead_data, labels
    

class BaseECGDatasetSubset(Dataset):
    def __init__(self, 
                 dataset, 
                 data_type: str,
                 transform: Optional[Tuple[Callable, ...]]=None,
                 random_seed: int=666):
        self.dataset = dataset
        self.data_type = data_type
        self.transform = transform
        assert self.data_type in ["train", "valid", "test"]
        
        # copying dataset attributes
        self.target_attr = self.dataset.target_attr
        self.indices = self.dataset.split_indices_dict[self.data_type]
        self.ecg_data = self.dataset.ecg_data.iloc[self.indices]
        
        self.random_seed = random_seed

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        lead_data, label = self.dataset[self.indices[int(idx)]]

        if self.transform is not None:
            for t in self.transform:
                lead_data = t(lead_data)

        return lead_data, label
