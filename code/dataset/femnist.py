import json
import os
import random
import typing
from collections import OrderedDict
from typing import Tuple, List

import numpy as np
from torch.utils.data import Dataset
import os.path as osp

import dataset
import widgets
from widgets import load_json


class FemnistDataset(Dataset):
    def __init__(self, data_index, data_dir):
        super().__init__()
        self.data_index = data_index
        self.data_dir = data_dir

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, item):
        file_name, label = self.data_index[item]
        if isinstance(self.data_dir, str):
            file_path = osp.join(self.data_dir, file_name)
            with open(file_path, 'r') as f:
                inputs = np.array(json.load(f), dtype=np.float32)
        else:
            index_in_full_data = int(file_name.split('.')[0])
            inputs = self.data_dir[index_in_full_data]
        inputs = inputs.reshape((1, 28, 28))
        inputs = inputs - 0.5
        return inputs, label

    @staticmethod
    def get_full_dataset(full_index_path, full_data_path) -> "FemnistDataset":
        """Return a full dataset, and the index is the same as the file name

        full_index should be a dict {"train": ... "test": ...}, whose values will be concatenated and sorted by int(ID)

        Args:
            full_index_path (str): index file path for full dataset
            full_data_path (str): data file path for full dataset

        Returns:
            FemnistDataset: the full dataset

        Notes:
            line number should be the same as file name, and continuous
        """
        import config

        print("loading full data...")
        full_data = np.load(full_data_path, allow_pickle=True)
        print("loaded")

        full_index = load_json(full_index_path)
        full_index = full_index['train'] + full_index['test']
        full_index.sort(key=lambda x: int(x[0].split('.')[0]))
        assert len(full_index) == int(full_index[-1][0].split('.')[0]) + 1, "index is not continuous"

        dataset_full = FemnistDataset(data_dir=full_data, data_index=full_index)
        return dataset_full


def get_user_indices(data_dir: str, *, train_ratio: float, shuffle=True) -> Tuple[
    typing.OrderedDict[str, List[int]], typing.OrderedDict[str, List[int]], typing.OrderedDict[str, List[int]]
]:
    """return user index in the train val test sets

    Args:
        data_dir: path to the user index
        train_ratio: the ratio of train set in train-val split
        shuffle: whether to shuffle the data before split

    Returns:
        user_train_indices, user_val_indices, user_test_indices: the indices of train, val, test set for each user
        - key: user ID
        - value: list of indices (int(file_name))
        - order: user ID
    """
    randomer = random.Random(0)
    user_array = os.listdir(data_dir)
    user_array.sort()
    user_indices = OrderedDict([
        (user_file_name.split('.')[0],
         widgets.load_json(osp.join(data_dir, user_file_name)))
        for user_file_name in user_array
    ])
    user_train_indices = OrderedDict()
    user_val_indices = OrderedDict()
    user_test_indices = OrderedDict()
    for user_name, user_index in user_indices.items():
        for key, val in user_index.items():
            new_val = [
                int(file_name.split('.')[0])
                for file_name, label in val
            ]
            user_index[key] = new_val
        train_index, val_index = dataset.train_val_split(
            user_index['train'], train_ratio=train_ratio, shuffle=shuffle,
            randomer=randomer
        )
        test_index = user_index['test']
        user_train_indices[user_name] = train_index
        user_val_indices[user_name] = val_index
        user_test_indices[user_name] = test_index

    return user_train_indices, user_val_indices, user_test_indices
