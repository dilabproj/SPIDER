from typing import Any, Tuple, List, Dict, Sequence, Optional, Callable, Union, ClassVar

import os
import pandas as pd
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer

from torch.utils.data import Dataset


class CPSCDataset(Dataset):
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
        super(CPSCDataset, self).__init__()
        
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
        Y = pd.read_csv(os.path.join(self.db_dir, "total_df.csv"))
        
        # label_group = {"conditions": rhythm_df['Acronym Name'].values,
        #                "rhythm": cond_df['Acronym Name'].values}
        
        diagnostics = Y['Labels'].apply(eval)
        mlb = MultiLabelBinarizer()
        ecg_label = pd.DataFrame(mlb.fit_transform(diagnostics), index=diagnostics.index, columns=mlb.classes_)
        
        # removing labels of only 1 sample
        ecg_label = ecg_label.loc[:, ecg_label.sum(axis=0) > 100]
            
        return Y, ecg_label

    def _split(self):
        # cv_split number 
        train_ratio, valid_ratio, test_ratio = self.split_ratio
        strat_id = np.zeros(self.__len__())
        
        train_id, valid_id, test_id = [[], [], []]
        
        train_valid_id = (~self.ecg_data['is_validation']).values.nonzero()[0]
        test_id = (self.ecg_data['is_validation']).values.nonzero()[0]

        if self.cv_split is None:
            validintrain_ratio = valid_ratio / (train_ratio + valid_ratio)
            sss = StratifiedShuffleSplit(n_splits=1, test_size=validintrain_ratio, random_state=self.random_seed)
            train_idx, valid_idx = next(sss.split(strat_id[train_valid_id], strat_id[train_valid_id]))
        else:
            cv_cnt, cv_id = self.cv_split
            logger.info("Spliting Training and Validation with Cross-Validation: Fold %s / %s", cv_id, cv_cnt)
            skf = StratifiedKFold(n_splits=cv_cnt, shuffle=True, random_state=self.random_seed)
            kfs = list(skf.split(strat_id[train_valid_id], strat_id[train_valid_id]))
            train_idx, valid_idx = kfs[cv_id - 1]
        
        train_id = train_valid_id[train_idx]
        valid_id = train_valid_id[valid_idx]
        
        assert set(train_id) | set(valid_id) | set(test_id) == set(range(self.__len__()))
        assert set(train_id) & set(valid_id) == set()
        assert set(train_id) & set(test_id) == set()
        assert set(valid_id) & set(test_id) == set()
        
        return {"train": train_id, "valid": valid_id, "test": test_id}
        
    def __len__(self):
        return self.ecg_data.shape[0]

    def __getitem__(self, idx):
        row = self.ecg_data.iloc[idx]
        labels = self.ecg_label.iloc[idx]
        
        path = row['Path']
        lead_data = loadmat(path)['ECG'][0, 0][2]
        
        if self.preprocess_lead is not None:
            lead_data = [self.preprocess_lead(lead_data[n], **self.preprocess_args) for n in range(lead_data.shape[0])]
        
        lead_data = np.array(lead_data, dtype='float32')
        if lead_data.shape[-1] > 5000:  # longer than 10s
            lead_data = lead_data[:, : 5000]
        elif lead_data.shape[-1] < 5000:  # shorter than 10s
            pad_len = 5000 - lead_data.shape[-1]
            lead_data = np.pad(lead_data, ((0, 0), (0, pad_len)), 'constant', constant_values=(0, 0))
        
        if self.sampling_rate == 100:
            lead_data = lead_data[:, ::5]
        
        if self.transform is not None:
            lead_data = Compose(self.transform)(lead_data)
            
        if self.timestep_first:
            lead_data = lead_data.T
        
        return lead_data, labels.values


class CPSCDatasetSubset(Dataset):
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
        
        self.random_seed = random_seed

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        lead_data, label = self.dataset[self.indices[int(idx)]]

        if self.transform is not None:
            lead_data = Compose(self.transform)(lead_data)

        return lead_data, label