import random
import typing
from collections import OrderedDict
from typing import Union, List, Tuple

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

import config
import os.path as osp
from . import utils


def _load_csv(path: str, columns):
    file_index = pd.read_csv(path)
    file_index = file_index[columns]
    file_index = file_index.values
    file_index = file_index.tolist()
    return file_index


class INaturalistDataset(Dataset):
    """iNaturalist dataset
    """
    def __init__(self, file_index: Union[str, List[Tuple[str, int]]], root=config.DATA_INATURALIST_DIR,
                 *, transform=None):
        """
        Args:
            file_index: list of file paths and labels, or path to a csv file, each row has two entries:
                - path from root to the image
                - label
            root: root directory of dataset
            transform: transform to apply on image
        """
        super().__init__()
        self.root = root
        self.transform = transform
        if isinstance(file_index, str):
            file_index = _load_csv(file_index, columns=['data_path', 'label_id'])
            self.file_index = file_index
        else:
            self.file_index = file_index

    def __getitem__(self, index):
        image_name, target = self.file_index[index]
        with open(osp.join(self.root, image_name), 'rb') as f:
            x = Image.open(f)
            x = x.convert('RGB')

        if self.transform is not None:
            x = self.transform(x)

        return x, target

    def __len__(self):
        return len(self.file_index)


def get_user_indices(
        train_ratio: float = None
) -> Union[
    Tuple[typing.OrderedDict[int, List[int]], typing.OrderedDict[int, List[int]], typing.OrderedDict[int, List[int]]],
    Tuple[typing.OrderedDict[int, List[int]], typing.OrderedDict[int, List[int]]]
]:
    randomer = random.Random(0)

    def get_map(path):
        # row_id is consistent with the dataset
        client_ids = _load_csv(path, columns='client_id')
        user_index = dict()
        for row_id, client_id in enumerate(client_ids):
            user_index.setdefault(client_id, []).append(row_id)
        user_index = OrderedDict(sorted(user_index.items(), key=lambda x: x[0]))
        return user_index

    user_train_val_indices = get_map(config.DATA_INATURALIST_TRAIN_INDEX_PATH)
    user_test_indices = get_map(config.DATA_INATURALIST_TEST_INDEX_PATH)
    if train_ratio is not None:
        users = list(user_train_val_indices.keys())
        users_train, users_val = utils.train_val_split(users, train_ratio=train_ratio, shuffle=True, randomer=randomer)
        users_train = set(users_train)
        users_val = set(users_val)
        user_train_indices = OrderedDict(
            (key, val) for key, val in user_train_val_indices.items() if key in users_train
        )
        user_val_indices = OrderedDict(
            (key, val) for key, val in user_train_val_indices.items() if key in users_val
        )
        return user_train_indices, user_val_indices, user_test_indices
    else:
        return user_train_val_indices, user_test_indices
