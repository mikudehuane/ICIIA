import logging
import random
from copy import deepcopy
from typing import List, Dict, Tuple, Optional, Union, Callable, Any

import torch
from torch.utils.data import Sampler

_logger = logging.getLogger(__name__)


class UserBatchSampler(Sampler):
    """generate batch per user

    Attributes:
        get_user_order: function that will be called in the beginning of __iter__,
            taken the number of users
            return the order of users
    """

    def __init__(self, user_indices: Union[Dict[Any, List[int]],
                                           List[List[int]]],
                 *, shuffle=False, batch_size: int = 0, random_batch_size=False):
        super().__init__(None)
        self.shuffle = shuffle
        self.random_batch_size = random_batch_size
        if isinstance(user_indices, Dict):
            self.user_indices = sorted(user_indices.items(), key=lambda x: x[0])
        else:
            self.user_indices = list(enumerate(user_indices))
        self.batch_size = batch_size
        assert isinstance(self.batch_size, int) and self.batch_size >= 0, "batch_size should be a non-negative integer"

        if self.batch_size != 0:
            self.length = 0
            for user_name, user_data_index in self.user_indices:
                self.length += (len(user_data_index) - 1) // self.batch_size + 1
            if self.random_batch_size:
                _logger.warning('random_batch_size results in not accurate length of the batch sampler')
                self.length *= 2
        else:
            assert not self.random_batch_size, "random_batch_size should be False when batch_size is 0"
            self.length = len(self.user_indices)

        self.get_user_order = None

    def __iter__(self):
        if self.get_user_order is None:
            if self.shuffle:
                user_order = torch.randperm(len(self.user_indices)).tolist()
            else:
                user_order = list(range(len(self.user_indices)))
        else:
            user_order = self.get_user_order(len(self.user_indices))

        for idx in user_order:
            user_name, user_data_index = self.user_indices[idx]
            row_ids = user_data_index
            if self.batch_size == 0:
                yield row_ids
            else:
                start_idx = 0
                while start_idx < len(row_ids):
                    if self.random_batch_size:
                        if random.random() < 0.5:
                            batch_size = 1
                        else:
                            batch_size = self.batch_size
                    else:
                        batch_size = self.batch_size
                    end_idx = min(start_idx + batch_size, len(row_ids))
                    yield row_ids[start_idx: end_idx]
                    start_idx = end_idx

    def register_set_user_order_func(self, func):
        self.get_user_order = func

    def __len__(self):
        return self.length


def set_user_order_randomly_func(seed=None) -> Callable[[int], List[int]]:
    """return a function that set the order of users randomly

    Args:
        seed: global random seed, ensures the following results:
            - the returned function generates different sequence in each call
            - the returned function generates the same sequence across different script runs
    """
    if seed is None:
        randomer = random
    else:
        randomer = random.Random(seed)

    def func(num_users: int) -> List[int]:
        user_order = list(range(num_users))
        randomer.shuffle(user_order)
        return user_order

    return func
