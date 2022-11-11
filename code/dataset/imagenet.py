import random
from collections import OrderedDict
from copy import deepcopy
from typing import Tuple, List, Dict
import typing
import os.path as osp

import numpy as np
from torch.utils.data import Dataset

import config
import pandas as pd

from . import utils

_SHEET_NAME_GROUPS = 'groups'
_SHEET_NAME_INFO = 'correspondence'
_DEFAULT_ANNOTATION_PATH = osp.join(config.CONFIG_DIR, 'imagenet_class_group.xlsx')


def parse_superclass_annotation(
        annotation_file_path: str = _DEFAULT_ANNOTATION_PATH
) -> Tuple[List[int], List[Tuple[List[int], str]]]:
    class_groups = pd.read_excel(
        annotation_file_path, sheet_name=_SHEET_NAME_GROUPS,
        dtype={'super_class': str, 'class_name': str, 'wnid': str,
               'annotated_super_class': np.int64, 'annotated_super_class_name': str},
        engine='openpyxl'
    )
    class_info = pd.read_excel(
        annotation_file_path, sheet_name=_SHEET_NAME_INFO,
        header=None, dtype=str, engine='openpyxl'
    )

    sub2super = class_groups.annotated_super_class.to_list()
    sub2super_name = class_groups.annotated_super_class_name.to_list()
    superclass_names = class_info[1].to_list()

    super2sub = [([], superclass_name) for superclass_name in superclass_names]
    for subclass_idx, superclass_idx in enumerate(sub2super):
        super2sub[superclass_idx][0].append(subclass_idx)
        assert sub2super_name[subclass_idx] == super2sub[superclass_idx][
            1], 'subclass {} has inconsistent superclass'.format(subclass_idx)

    return sub2super, super2sub


def get_user_split(
        user_indices_train_val: typing.OrderedDict[int, List[int]],
        user_indices_test: typing.OrderedDict[int, List[int]]
) -> Tuple[typing.OrderedDict[str, List[int]], typing.OrderedDict[str, List[int]]]:
    assert len(user_indices_train_val) == len(user_indices_test), 'train_val and test should have same number of users'

    user_indices_train_val = deepcopy(user_indices_train_val)
    user_indices_test = deepcopy(user_indices_test)

    num_train_mult = 80
    num_test_mult = 16

    new_user_indices_train_val = OrderedDict()
    new_user_indices_test = OrderedDict()

    cls_format = '{:0' + str(len(str(len(user_indices_train_val)))) + 'd}'  # 父类 ID 的格式
    key_format = cls_format + '-{:05d}'  # 不会超过五位

    for (user_index_tv_key, user_index_tv), (user_index_test_key, user_index_test) in zip(
            user_indices_train_val.items(), user_indices_test.items()):
        assert user_index_tv_key == user_index_test_key, 'train_val and test should have same user index'
        # know the number of groups
        num_groups_tv = (len(user_index_tv) + num_train_mult - 1) // num_train_mult
        num_groups_test = (len(user_index_test) + num_test_mult - 1) // num_test_mult

        # "large" for the dataset with more *groups*
        if num_groups_tv > num_groups_test:
            new_user_indices_small = new_user_indices_test
            new_user_indices_large = new_user_indices_train_val
            num_groups_small = num_groups_test
            num_groups_large = num_groups_tv
            num_mult_large = num_train_mult
            num_mult_small = num_test_mult
            user_index_large = user_index_tv
            user_index_small = user_index_test
        else:
            new_user_indices_small = new_user_indices_train_val
            new_user_indices_large = new_user_indices_test
            num_groups_small = num_groups_tv
            num_groups_large = num_groups_test
            num_mult_large = num_test_mult
            num_mult_small = num_train_mult
            user_index_large = user_index_test
            user_index_small = user_index_tv

        # tv/test, choose the min as the base
        for sub_user_idx in range(num_groups_small):
            large_range_min = (num_groups_large * sub_user_idx) // num_groups_small
            large_range_max = (num_groups_large * (sub_user_idx + 1)) // num_groups_small
            assert large_range_max > large_range_min, 'large_range_max should be larger than large_range_min'
            large_range_min = large_range_min * num_mult_large
            large_range_max = large_range_max * num_mult_large
            key = key_format.format(user_index_tv_key, sub_user_idx)
            new_user_indices_large[key] = user_index_large[large_range_min: large_range_max]
            key = key_format.format(user_index_test_key, sub_user_idx)
            new_user_indices_small[key] = user_index_small[sub_user_idx * num_mult_small:
                                                           (sub_user_idx + 1) * num_mult_small]

    return new_user_indices_train_val, new_user_indices_test


def _to_soft(user_indices: typing.OrderedDict[int, List[int]], randomer: random.Random, *, reserve_ratio=0.5
             ) -> typing.OrderedDict[int, List[int]]:
    user_indices = deepcopy(user_indices)

    orig_num_samples = OrderedDict(
        ((key, len(value)) for key, value in user_indices.items())
    )

    random_pool = []
    new_user_indices = OrderedDict()
    for uid, index in user_indices.items():
        reserve_count = int(len(index) * reserve_ratio)
        new_user_indices[uid] = index[:reserve_count]
        random_pool.extend(index[reserve_count:])

    randomer.shuffle(random_pool)
    start = 0
    for uid, num_samples in orig_num_samples.items():
        end = start + num_samples - len(new_user_indices[uid])
        new_user_indices[uid].extend(random_pool[start: end])
        start = end

    return new_user_indices


def get_final_user_split(
        train_val_dataset, test_dataset, *, train_ratio: float, soft_reserve_ratio=0.5,
        sub_super_map=None
) -> Tuple[typing.OrderedDict[str, List[int]], typing.OrderedDict[str, List[int]], typing.OrderedDict[str, List[int]]]:
    if sub_super_map is None:
        imagenet_sub2super, imagenet_super2sub = parse_superclass_annotation()
    else:
        imagenet_sub2super, imagenet_super2sub = sub_super_map

    randomer = random.Random(0)
    user_train_val_indices = utils.get_class_groups(
        train_val_dataset, imagenet_sub2super, shuffle_within_group=True, randomer=randomer)
    user_test_indices = utils.get_class_groups(
        test_dataset, imagenet_sub2super, shuffle_within_group=True, randomer=randomer)
    if soft_reserve_ratio < 1.0:
        assert soft_reserve_ratio >= 0.0, 'soft_reserve_ratio should be in [0, 1]'
        user_train_val_indices = _to_soft(user_train_val_indices, randomer=randomer, reserve_ratio=soft_reserve_ratio)
        user_test_indices = _to_soft(user_test_indices, randomer=randomer, reserve_ratio=soft_reserve_ratio)
    else:
        assert soft_reserve_ratio == 1.0, 'soft_reserve_ratio should be in [0, 1]'
    user_train_val_indices, user_test_indices = get_user_split(
        user_train_val_indices, user_test_indices)

    user_train_indices = OrderedDict()
    user_val_indices = OrderedDict()
    for user_id in user_train_val_indices:
        train_val_indices = user_train_val_indices[user_id]
        train_index, val_index = utils.train_val_split(
            train_val_indices, train_ratio=train_ratio, shuffle=True,
            randomer=randomer
        )
        user_train_indices[user_id] = train_index
        user_val_indices[user_id] = val_index

    return user_train_indices, user_val_indices, user_test_indices
