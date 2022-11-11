import csv
import json
import random
import typing
from collections import OrderedDict
from copy import deepcopy
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
import torchvision.datasets
from PIL import Image
from torch.utils.data import Dataset
import os.path as osp
import os

from torchvision import transforms

import config

from . import utils


CELEBA_ATTR_NAMES = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
ALL_CELEBA_ATTRS = deepcopy(CELEBA_ATTR_NAMES)
ALL_CELEBA_ATTRS.remove('Male')
ALL_CELEBA_ATTRS.remove('Young')


class CelebaDataset(Dataset):
    """LEAF Celeba dataset

    Attributes:
        user_index (OrderedDict[str, List[int]]): processed user index, mapping from user id to a list of image indices.
            Note that the passed user index (to __init__) is a map to a list of image names, not indices.
            Ordered by int(uid).
        images_name_label (List[Tuple[str, int]]): list of image names and labels, self[item] will fetch
            images_name_label[item].
            Ordered by directly sorting image names
        images_name2index (Dict[str, int]): reverse map from image name to index
    """
    def __init__(self, user_index_path: str,
                 image_folder_path=osp.join(config.DATA_CELEBA_ROOT_DIR, 'raw', 'img_align_celeba'),
                 transform=None, target_attr: str = 'Smiling'):
        """Initialize the dataset

        all images are specified by user_index_path (dict values)

        Args:
            user_index_path: path to the user index file,
                e.g. osp.join(config.DATA_CELEBA_ROOT_DIR, 'raw', 'train', 'all_data_niid_0_keep_5_train_9.json')
            image_folder_path: path to the image folder
            transform: transform to apply to the image
            target_attr: target attribute, e.g. 'Smiling', to be predicted
                - if as 'all', produce a 40-dim tensor of all attrs as target
        """
        self.image_folder_path = image_folder_path
        self.transform = transform
        self.target_attr = target_attr

        # self.attr[i] is the attribute of image f"{i}.jpg" (0 filled)
        self.attr_names, self.attrs = self._load_attrs(config.DATA_CELEBA_ATTR_PATH)
        self.attrs_frame = pd.DataFrame(self.attrs, columns=self.attr_names)
        self.attrs_np = np.array(self.attrs)

        # load the index dict
        with open(user_index_path, 'r') as f:
            user_index = json.load(f)
        user_index = user_index['user_data']
        # global index for the dataset
        self.images_name_label = [
            (image_name, image_label)
            for uid, user_content in user_index.items()
            for image_name, image_label in zip(user_content['x'], user_content['y'])
        ]
        self.images_name_label.sort(key=lambda x: x[0])
        # self._check_consistent()  # check consistentcy with attrs_frame
        # label from float to int
        # check for label 0.0 or 1.0
        for i, (_, label) in enumerate(self.images_name_label):
            assert label in (0.0, 1.0), f'invalid label {label} at index {i}'
        self.images_name_label = [(image_name, int(image_label + 1e-7)) for image_name, image_label in self.images_name_label]
        # reverse map from image name to index
        self.images_name2index = {image_name: index for index, (image_name, _) in enumerate(self.images_name_label)}
        # generate the user index (used by batch sampler)
        # sort the user_index according to uid
        user_index = list(user_index.items())
        user_index.sort(key=lambda x: int(x[0]))
        self.user_index = OrderedDict()
        for uid, user_content in user_index:
            self.user_index[uid] = [self.images_name2index[image_name] for image_name in user_content['x']]

    def __len__(self):
        return len(self.images_name_label)

    def __getitem__(self, item: int):
        """

        Args:
            item: index from 0

        Returns:
            (image_data, target)
        """
        image_name, _ = self.images_name_label[item]
        if self.target_attr == 'all':
            row_index = int(image_name.split('.')[0])
            target = self.attrs_frame.loc[row_index, ALL_CELEBA_ATTRS].values
            target = ((target + 1) // 2).astype(np.float32)
        else:
            target = self.get_attr(image_name, self.target_attr)

        with open(osp.join(self.image_folder_path, image_name), 'rb') as f:
            x = Image.open(f)
            x = x.convert('RGB')

        if self.transform is not None:
            x = self.transform(x)

        return x, target

    def get_train_val_user_index_split(self, train_ratio: float, shuffle=True) -> Tuple[
        typing.OrderedDict[str, List[int]], typing.OrderedDict[str, List[int]]
    ]:
        """Split the user_index into train and validation set

        Args:
            train_ratio: ratio of train set
            shuffle: whether to shuffle the user index before splitting
        """
        randomer = random.Random(0)

        user_train_indices = OrderedDict()
        user_val_indices = OrderedDict()

        self_user_index = deepcopy(self.user_index)
        for user_name, user_index in self_user_index.items():
            train_index, val_index = utils.train_val_split(
                user_index, train_ratio=train_ratio, shuffle=shuffle,
                randomer=randomer
            )
            user_train_indices[user_name] = train_index
            user_val_indices[user_name] = val_index

        return user_train_indices, user_val_indices

    @staticmethod
    def _load_attrs(filepath):
        attrs = [[0] * 40]  # 0 reserved to ensure line number is the same as image name

        with open(filepath) as csv_file:
            csv_file.readline()
            attr_names = csv_file.readline().strip().split()
            while True:
                line = csv_file.readline().strip()
                if line == '':
                    break
                attr = line.split()
                assert len(attrs) == int(attr[0].split('.')[0]), f'line number {len(attrs)} != image name {attr[0]}'
                attrs.append([int(x) for x in attr[1:]])

        return attr_names, attrs

    def _check_consistent(self):
        for image_name, label in self.images_name_label:
            attr = self.get_attr(image_name, 'Smiling')
            assert attr == label, f'attr {attr} != label {label} for image {image_name}'

    def get_attr(self, image_name, attr_name):
        attr = self.attrs_frame.loc[int(image_name.split('.')[0]), attr_name]
        if attr == -1:
            attr += 1
        return attr

    def statistic_users_attr(self):
        """count the attr-user distribution and print the results


        Returns:
            a table, 1st column is uid, the remaining columns are the number of classes for each attr
        """
        columns = ['uid'] + self.attr_names
        data_class_count = []
        data_posi_count = []
        for uid, indices in self.user_index.items():
            user_attr = self.attrs_frame.loc[
                [int(self.images_name_label[index][0].split('.')[0])
                 for index in indices]
            ].to_numpy()
            num_classes = [len(np.unique(user_attr[:, column_idx])) for column_idx in range(user_attr.shape[1])]
            posi_count = [int(np.sum(user_attr[:, column_idx] == 1)) for column_idx in range(user_attr.shape[1])]
            data_class_count.append([uid] + num_classes)
            data_posi_count.append([uid] + posi_count)
        data_class_count_frame = pd.DataFrame(data_class_count, columns=columns)
        print(data_class_count_frame)
        # number of users that have 2 classes
        data_class_count_sum = np.array([cnt[1:] for cnt in data_class_count])
        data_class_count_sum = np.sum((data_class_count_sum == 2).astype(np.int64), axis=0, keepdims=True)
        data_class_count_sum_frame = pd.DataFrame(data_class_count_sum, columns=columns[1:])
        print(data_class_count_sum_frame)
        data_posi_count = pd.DataFrame(data_posi_count, columns=columns)
        print(data_posi_count)
        # print the sum of each column
        attrs_sum = (self.attrs_frame == 1).sum(axis=0) / self.attrs_frame.shape[0]
        print(attrs_sum)


def get_user_indices(
        train_ratio: float, shuffle=True,
        user_index_path_train=config.DATA_CELEBA_TRAIN_INDEX_PATH,
        user_index_path_test=config.DATA_CELEBA_TEST_INDEX_PATH,
        split_method='sample'
):
    """return user index in the train val test sets

        Args:
            train_ratio: the ratio of train set in train-val split
            shuffle: whether to shuffle the data before split
            user_index_path_train: path to the train user index file
            user_index_path_test: path to the test user index file
            split_method: train-val split by users, or directly by samples

        Returns:
            user_train_indices, user_val_indices, user_test_indices: the indices of train, val, test set for each user
            - key: user ID
            - value: list of indices (int(file_name))
            - order: int(user ID)
        """
    randomer = random.Random(0)
    dataset_train_val = CelebaDataset(
        user_index_path=user_index_path_train, transform=None)
    dataset_test = CelebaDataset(user_index_path=user_index_path_test, transform=None)

    user_train_indices = OrderedDict()
    user_val_indices = OrderedDict()
    if split_method == 'sample':
        for user_name, user_index in dataset_train_val.user_index.items():
            train_index, val_index = utils.train_val_split(
                user_index, train_ratio=train_ratio, shuffle=shuffle,
                randomer=randomer
            )
            user_train_indices[user_name] = train_index
            user_val_indices[user_name] = val_index
    elif split_method == 'user':
        users_train_val = list(dataset_train_val.user_index.keys())
        users_train_val.sort()
        randomer.shuffle(users_train_val)
        users_train, users_val = utils.train_val_split(users_train_val, train_ratio=train_ratio, shuffle=shuffle,
                                                       randomer=randomer)
        for user_name in users_train:
            user_train_indices[user_name] = dataset_train_val.user_index[user_name]
        for user_name in users_val:
            user_val_indices[user_name] = dataset_train_val.user_index[user_name]
    else:
        raise ValueError(f'unknown split method {split_method}')

    return user_train_indices, user_val_indices, dataset_test.user_index


def get_celeba_all_labels(all_labels: torch.Tensor) -> torch.Tensor:
    """given all attrs, fetch only the ALL_CELEBA_ATTRS"""
    all_labels = all_labels.to(dtype=torch.float32)
    all_labels = pd.DataFrame(all_labels.numpy(), columns=CELEBA_ATTR_NAMES)
    all_labels = torch.from_numpy(all_labels.loc[:, ALL_CELEBA_ATTRS].values)
    return all_labels
