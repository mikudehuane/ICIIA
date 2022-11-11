import random
import typing
from collections import OrderedDict
from copy import deepcopy
import os.path as osp
from typing import List, Dict, Tuple, Optional, Any, Union

import numpy as np
import torch
from torch.utils.data import Dataset


def train_val_split(full_index, *, train_ratio=0.8, shuffle=True, randomer=random):
    """Split dataset into train and validation sets.

    Args:
        full_index (list): list of indices
        train_ratio (float): ratio of train set
        shuffle (bool): whether to shuffle the dataset
        randomer (random.Random): random generator
    """
    if len(full_index) < 2:
        raise ValueError("The length of full_index must be greater than 1, to split")
    split_index = max(int(len(full_index) * train_ratio), 1)  # at least 1 to avoid empty trainset
    full_index = deepcopy(full_index)  # avoid modify full_index
    if shuffle:
        randomer.shuffle(full_index)
    val_index = full_index[split_index:]
    train_index = full_index[:split_index]
    assert len(train_index) > 0, 'train_index is empty'
    assert len(val_index) > 0, 'val_index is empty'
    return train_index, val_index


class RemappedDataset(Dataset):
    def __init__(self, dataset: Dataset, mapping: List[int]):
        self.dataset = dataset
        self.mapping = mapping

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, index):
        return self.dataset[self.mapping[index]]


def load_features(dset_folder: str, need_features=True) -> Tuple[torch.Tensor, torch.Tensor]:
    """load data from dset_folder

    Args:
        dset_folder: parent folder
        need_features: whether to load features

    Returns:
        features, labels
    """
    print("loading labels from {}".format(dset_folder))
    loaded_labels = torch.from_numpy(np.loadtxt(osp.join(dset_folder, 'labels.txt'), dtype=np.int64, delimiter=','))
    if need_features:
        print("loading features from {}".format(dset_folder))
        loaded_features = torch.from_numpy(np.load(osp.join(dset_folder, 'features.npy')))
    else:
        loaded_features = torch.zeros((loaded_labels.shape[0], 0))
    return loaded_features, loaded_labels


def get_user_split(user_indices: List[List[int]], max_num_samples: Optional[int] = None,
                   *, shuffle=False, seed=0) -> List[List[int]]:
    if seed is None:
        randomer = random
    else:
        randomer = random.Random(seed)

    user_indices = deepcopy(user_indices)
    new_user_indices = []

    for user_index in user_indices:
        if shuffle:
            randomer.shuffle(user_index)

        if len(user_index) <= max_num_samples:
            new_user_indices.append(user_index)
        else:
            num_groups = (len(user_index) + max_num_samples - 1) // max_num_samples
            for i in range(num_groups):
                new_user_indices.append(user_index[i * max_num_samples: (i + 1) * max_num_samples])

    return new_user_indices


class DatasetWithUser(Dataset):
    def __init__(self, dataset: Dataset, user_indices: Union[Dict[Any, List[int]], List[int]]):
        self.dataset = dataset

        if isinstance(user_indices, Dict):
            user_indices = sorted(user_indices.items(), key=lambda x: x[0])
            user_indices = [x[1] for x in user_indices]
        # reverse mapping on user_indices
        self.user_indices_rev = dict()
        for uid, user_index in enumerate(user_indices):
            for index in user_index:
                assert index not in self.user_indices_rev, 'data index {} is duplicated'.format(index)
                self.user_indices_rev[index] = uid

    def __len__(self):
        return len(self.user_indices_rev)

    def __getitem__(self, index):
        if index not in self.user_indices_rev:
            raise RuntimeError('index {} is not in user_indices'.format(index))

        sample = self.dataset[index]
        uid = np.int64(self.user_indices_rev[index])

        assert isinstance(sample, tuple), 'sample must be a tuple'
        return sample + (uid,)


def get_class_groups(
        original_dataset: Dataset, sub2super: List[int], *, shuffle_within_group=False, randomer=None
) -> typing.OrderedDict[int, List[int]]:
    if randomer is None:
        randomer = random
    super2index: Dict[int, List[int]] = dict()
    for sample_idx, sample in enumerate(original_dataset):
        label = sample[-1]
        super_label = sub2super[label]
        if super_label not in super2index:
            super2index[super_label] = []
        super2index[super_label].append(sample_idx)
    super_and_index = list(super2index.items())
    super_and_index.sort(key=lambda x: x[0])
    groups = OrderedDict()
    for super_label, index_list in super_and_index:
        if shuffle_within_group:
            randomer.shuffle(index_list)
        groups[super_label] = index_list
    return groups
