import os, sys, random, ast, math

from typing import Any, Tuple, List, Dict, Sequence, Optional, Callable, Union, ClassVar

import pandas as pd
import numpy as np
import wfdb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

import torch
from torch.utils.data import Dataset, sampler


class PTBXLDataset(Dataset):
    def __init__(self, 
                 db_dir, 
                 sampling_rate=500,
                 target_attr=None,
                 split_ratio: Tuple[float, float, float]=(0.7, 0.1, 0.2),
                 cv_split: Optional[Tuple[int, int]] = None,
                 preprocess_lead: Optional[Callable] = None,
                 preprocess_args: Optional[Dict[str, Any]] = None,
                 timestep_first=False,
                 transform: Optional[Tuple[Callable, ...]] = None,
                 random_seed: int = 666):
        super(PTBXLDataset, self).__init__()
        
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
        
        self.ecg_data, self.ecg_label = self._load_data()
        self.split_indices_dict = self._split()
        
    def _load_data(self):
        agg_df = pd.read_csv(os.path.join(self.db_dir, "scp_statements.csv"), index_col=0)
        label_group = {"diagnostic": agg_df[agg_df['diagnostic'] == 1].index.values, 
                       "form": agg_df[agg_df['form'] == 1].index.values, 
                       "rhythm": agg_df[agg_df['rhythm'] == 1].index.values}
        
        all_labels = agg_df.index.values
        if self.target_attr is not None:
            assert self.target_attr in all_labels, "No such target attribute"
        
        Y = pd.read_csv(os.path.join(self.db_dir, "ptbxl_database.csv"), index_col="ecg_id")

        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
        Y.scp_codes = Y.scp_codes.apply(lambda x: [k for k in x])
        
        diagnostics = Y.scp_codes
        
        mlb = MultiLabelBinarizer()
        ecg_label = pd.DataFrame(mlb.fit_transform(diagnostics), index=diagnostics.index, columns=mlb.classes_)
        ecg_label = ecg_label.loc[:, ecg_label.sum(axis=0) > 0]
            
        return Y, ecg_label
    
    def _split(self):
        # strat 9, 10 as the test set
        # cv_split number 
        train_ratio, valid_ratio, test_ratio = self.split_ratio
        # strat_id = self.ecg_data['strat_fold']
        
        train_id, valid_id, test_id = [[], [], []]
        if self.cv_split is None:
            cv_cnt, cv_id = 10, 1
        else:
            cv_cnt, cv_id = self.cv_split

        train_valid_id = (self.ecg_data['strat_fold']!=cv_id).values.nonzero()[0].tolist()
        if valid_ratio == 0:
            train_id = train_valid_id
            valid_id = []
        else:
            train_id, valid_id = train_test_split(train_valid_id, 
                                                  test_size=valid_ratio / (valid_ratio + train_ratio), 
                                                  random_state=self.random_seed)
            
        test_id = (self.ecg_data['strat_fold']==cv_id).values.nonzero()[0].tolist()
        
        assert set(train_id) | set(valid_id) | set(test_id) == set(range(self.__len__()))
        assert set(train_id) & set(valid_id) == set()
        assert set(train_id) & set(test_id) == set()
        assert set(valid_id) & set(test_id) == set()
        
        return {"train": train_id, "valid": valid_id, "test": test_id}
        
    def __len__(self):
        return self.ecg_data.shape[0]

    def __getitem__(self, idx):
        row = self.ecg_data.iloc[idx]
        label = self.ecg_label.iloc[idx]
        
        if self.sampling_rate == 100:
            path = os.path.join(self.db_dir, row['filename_lr'])
        else:  # self.sampling_rate == 500
            path = os.path.join(self.db_dir, row['filename_hr'])
        
        lead_data, meta = wfdb.rdsamp(path)
        lead_data = lead_data.T
        
        if self.preprocess_lead is not None:
            lead_data = [self.preprocess_lead(lead_data[n], **self.preprocess_args) for n in range(lead_data.shape[0])]
        
        lead_data = np.array(lead_data, dtype='float32')    
        
        if self.transform is not None:
            for t in self.transform:
                lead_data = t(lead_data)
            
        if self.timestep_first:
            lead_data = lead_data.T
        
        return lead_data, label.values


class PTBXLDatasetSubset(Dataset):
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
        self.ecg_label = self.dataset.ecg_label.iloc[self.indices]
        self.ecg_label = self.dataset.ecg_label.iloc[self.indices]
        
        self.random_seed = random_seed

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        lead_data, label = self.dataset[self.indices[int(idx)]]

        if self.transform is not None:
            for t in self.transform:
                lead_data = t(lead_data)

        return lead_data, label


class ImbalancedDatasetSampler(sampler.Sampler):
    def __init__(self,  # pylint: disable=super-init-not-called
                 dataset: Union[PTBXLDataset, PTBXLDatasetSubset],
                 weight_function: Callable,
                 num_samples: Optional[int] = None):
        self.dataset = dataset
        self.weight_function = weight_function

        # Get distribution of target
        self.data_uni, self.data_cnt, self.weight_mapping = self.get_weight(self.dataset, self.weight_function)
        # logger.info("Weight Mapping: %s", self.weight_mapping)

        # Set weight for each sample
        targets = self.dataset.ecg_data[self.dataset.target_attr]
        self.weights = torch.tensor([self.weight_mapping[tar] for tar in targets], dtype=torch.float64)  # pylint: disable=not-callable

        # Save number of samples per iteration
        self.num_samples = num_samples if num_samples is not None else len(self.dataset)
        # logger.info("Weighted to total %s samples", self.num_samples)

    @staticmethod
    def get_weight(dataset: Union[PTBXLDataset, PTBXLDatasetSubset], weight_function: Callable):
        targets = dataset.ecg_data[dataset.target_attr]
        data_uni, data_cnt = np.unique(targets, return_counts=True)
        weight_mapping = {u: weight_function(c) for u, c in zip(data_uni, data_cnt)}
        return data_uni, data_cnt, weight_mapping

    @staticmethod
    def wf_one(x):  # pylint: disable=unused-argument
        return 1.0

    @staticmethod
    def wf_x(x):
        return x

    @staticmethod
    def wf_onedivx(x):
        return 1.0 / x

    @staticmethod
    def wf_logxdivx(x):
        return math.log(x) / x

    def __iter__(self):
        return (i for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
