import os.path
import random
import re
from typing import Union, Callable, List, Dict, Tuple
from collections import OrderedDict

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import os.path as osp
import config

from .. import utils as dset_utils

_DEFAULT_ANNOTATION_PATH = osp.join(config.CONFIG_DIR, 'ucf101_class_group.xlsx')
_SHEET_NAME_GROUPS = 'groups'


def get_final_user_split(train_val_dataset, test_dataset, *, train_ratio: float):
    sub2super, super2sub = parse_superclass_annotation()
    randomer = random.Random(0)
    user_train_val_indices = dset_utils.get_class_groups(
        train_val_dataset, sub2super, shuffle_within_group=True, randomer=randomer)
    user_test_indices = dset_utils.get_class_groups(
        test_dataset, sub2super, shuffle_within_group=True, randomer=randomer)

    user_train_indices = OrderedDict()
    user_val_indices = OrderedDict()
    # train_val_split here (ensures video groups are not shared)
    paths_to_index = {path: index for index, path in enumerate(train_val_dataset.video_index)}
    for puid, train_val_indices in user_train_val_indices.items():
        train_val_paths = [train_val_dataset.video_index[i] for i in train_val_indices]
        train_indices, val_indices = train_val_split(train_val_paths, train_ratio=train_ratio,
                                                     randomer=randomer)
        user_train_indices[puid] = [paths_to_index[path] for path in train_indices]
        user_val_indices[puid] = [paths_to_index[path] for path in val_indices]

    # further split the users
    mult = 16  # try to the make the number of train/val/test samples be multiple of `mult` for each user
    # list of train/val/test
    new_indices = tuple(OrderedDict() for _ in range(3))
    assert list(user_train_indices.keys()) == list(user_val_indices.keys()) == list(user_test_indices.keys())
    cls_format = '{:0' + str(len(str(len(user_train_val_indices)))) + 'd}'  # parent class id format
    key_format = cls_format + '-{:05d}'  # user id format
    for puid in user_train_indices.keys():
        all_indices = [user_train_indices[puid], user_val_indices[puid], user_test_indices[puid]]
        num_mults = [(len(indices) + mult - 1) // mult for indices in all_indices]
        num_sub_users = min(num_mults)
        for sub_uid in range(num_sub_users):
            uid = key_format.format(puid, sub_uid)
            for set_idx in range(len(new_indices)):
                range_min = num_mults[set_idx] * sub_uid // num_sub_users
                range_max = num_mults[set_idx] * (sub_uid + 1) // num_sub_users
                num_min_mult = range_min * mult
                num_max_mult = range_max * mult
                new_indices[set_idx][uid] = all_indices[set_idx][num_min_mult: num_max_mult]

    return new_indices


class UCF101Dataset(Dataset):
    """UCF101 dataset
    """

    def __init__(
            self, video_loader: Callable[[str], np.ndarray], root: str = config.DATA_UCF101_DIR,
            video_index: Union[str, List[str]] = config.DATA_UCF101_TRAIN_INDEX_PATH,
            class_index: Union[str, Dict[str, int]] = config.DATA_UCF101_CLASS_INDEX_PATH,
    ):
        """initialize the dataset

        Args:
            video_loader: loader that takes a video path and returns a numpy array of frames
            root: root directory of the dataset
            video_index: index file for the dataset, e.g. trainlist01.txt
                - each line with a path, optionally with a label, which is ignored (the label is inferred from the path)
                - or the content, [video1_path, video2_path, ...])
                - the path is the absolute path, that is, if given as List[str], root argument will be invalid
            class_index: index file for the classes, e.g. classInd.txt (or the content, class_name -> class index)
                the target value of the dataset is the class index - 1
        """
        self.video_loader = video_loader
        self.root = root

        # process the class index
        if isinstance(class_index, str):
            self.class_index = dict()
            with open(class_index, 'r') as f:
                for line in f:
                    class_idx, class_name = line.strip().split()
                    self.class_index[class_name.lower()] = int(class_idx)
        else:
            self.class_index = class_index
        self.class_index_reverse = dict()
        for class_name, class_idx in self.class_index.items():
            self.class_index_reverse[class_idx] = class_name
        assert len(self.class_index) == 101, 'class_index should have 101 classes'
        # process the split index
        if isinstance(video_index, str):
            self.video_index = []
            with open(video_index, 'r') as f:
                for line in f:
                    line_s = line.strip().split()
                    if len(line_s) == 1:
                        video_path = osp.join(self.root, line_s[0])
                    elif len(line_s) == 2:
                        f_rpath, class_idx = line_s
                        cls_from_path = self._get_class_from_path(f_rpath)
                        cls_from_file = self.class_index_reverse[int(class_idx)]
                        assert cls_from_file == cls_from_path, 'class index in file does not match the path'
                        video_path = osp.join(self.root, f_rpath)
                    else:
                        raise RuntimeError("invalid line: {}".format(line))
                    self.video_index.append(video_path)
        else:
            self.video_index = video_index

    def __getitem__(self, item):
        video_path = self.video_index[item]
        label = self._get_label_from_path(video_path)

        frames = self.video_loader(video_path)

        return frames, label

    def __len__(self):
        return len(self.video_index)

    @staticmethod
    def _get_class_from_path(video_path: str) -> str:
        return osp.basename(video_path).split('_')[1].lower()

    def _get_label_from_path(self, video_path):
        return self.class_index[self._get_class_from_path(video_path)] - 1


def _group_index(video_index: List[str]) -> List[List[str]]:
    """group videos by the belonging group"""
    groups = OrderedDict()  # avoid the randomness of dict
    name_pattern = re.compile(r'v_(\w+)_g(\d+)_c(\d+)\.avi')
    for video_path in video_index:
        video_name = os.path.basename(video_path)
        video_name_match = re.match(name_pattern, video_name)
        video_class = video_name_match.group(1)
        video_group = video_name_match.group(2)
        group_name = (video_class, video_group)
        if group_name not in groups:
            groups[group_name] = []
        groups[group_name].append(video_path)
    return list(groups.values())


def train_val_split(
        video_index: List[str], train_ratio: float = 0.8,
        randomer=None
) -> Tuple[List[str], List[str]]:
    """split the video index into train and validation set

    Args:
        video_index: the video index
        train_ratio: ratio of the training set
        randomer: randomer to use, if None, use `random.Random(0)`

    Notes:
        - the split is done by group, train and val videos will not come from the same group
        - given the same params, the split keeps the same by setting the same randomer

    Returns:
        train_index, val_index
    """
    if randomer is None:
        randomer = random.Random(0)
    groups = _group_index(video_index)

    train_groups, val_groups = dset_utils.train_val_split(
        groups, train_ratio=train_ratio, shuffle=True, randomer=randomer)
    train_index = [video_path for group in train_groups for video_path in group]
    val_index = [video_path for group in val_groups for video_path in group]

    return train_index, val_index


def parse_superclass_annotation(
        annotation_file_path: str = _DEFAULT_ANNOTATION_PATH
) -> Tuple[List[int], List[Tuple[List[int], str]]]:
    """parse the ucf101 superclass annotation

    Args:
        annotation_file_path: the path of the annotation file

    Returns: a tuple of
        - class_index -> super_class_index
        - super_class_index -> (class_indices, super_class_name)
    """
    class_groups = pd.read_excel(
        annotation_file_path, sheet_name=_SHEET_NAME_GROUPS,
        dtype={'class_idx': np.int64, 'class_name': str,
               'annotated_super_class': np.int64, 'annotated_super_class_name': str, 'super_class_names': str},
        engine='openpyxl'
    )

    sub2super = class_groups.annotated_super_class.to_list()
    sub2super_name = class_groups.annotated_super_class_name.to_list()
    superclass_names = class_groups.super_class_names.to_list()
    while isinstance(superclass_names[-1], float):
        superclass_names.pop()

    super2sub = [([], superclass_name) for superclass_name in superclass_names]
    for subclass_idx, superclass_idx in enumerate(sub2super):
        super2sub[superclass_idx][0].append(subclass_idx)
        assert sub2super_name[subclass_idx] == super2sub[superclass_idx][1], \
            'subclass {} has inconsistent superclass'.format(subclass_idx)

    return sub2super, super2sub
