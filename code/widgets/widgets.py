import json
import sys
from typing import List, Tuple

import config
import os.path as osp
import time
import os
import torch


def load_json(path):
    """load a json file from path"""
    with open(path, 'r') as f:
        return json.load(f)


def dump_json(obj, path):
    """dump a json file to path"""
    with open(path, 'w') as f:
        json.dump(obj, f)


def config_log_dir_and_write_args(name, args, **kwargs):
    log_pdir = osp.join(config.LOG_DIR, name)
    log_dir = osp.join(log_pdir, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir)
    config.config_log(open(osp.join(log_dir, 'running_log.txt'), 'w'), **kwargs)

    with open(osp.join(log_dir, "args.txt"), 'w') as f:
        f.write("command:\n")
        f.write("python " + " ".join(sys.argv) + "\n")
        f.write("args:\n")
        f.write(json.dumps(args.__dict__) + "\n")

    return log_dir


def parse_fp(inp):
    import config

    inp = inp.replace('{PROJECT_DIR}', config.PROJECT_DIR)
    inp = inp.replace('{LOG_DIR}', config.LOG_DIR)
    return inp


def get_num_pars(module: torch.nn.Module):
    total_num = sum(p.numel() for p in module.parameters())
    trainable_num = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total_num, trainable_num


def uniform_partition(range_min, range_max, num_partitions) -> List[Tuple[int, int]]:
    if num_partitions > range_max - range_min:
        raise ValueError('num_partitions must be less than number of items')
    if num_partitions < 1:
        raise ValueError('num_partitions must be greater than 0')
    if range_max <= range_min:
        raise ValueError('range_max must be greater than range_min')

    return [(range_min + i * (range_max - range_min) // num_partitions,
             range_min + (i + 1) * (range_max - range_min) // num_partitions)
            for i in range(num_partitions)]


def load_states(predictor, path):
    """load model params (fix prev issue of profile adding total_ops"""
    states = torch.load(path, map_location=config.DEVICE)
    print('load model from', path)
    for key in list(states.keys()):
        if key.endswith('total_ops') or key.endswith('total_params'):
            del states[key]
    predictor.load_state_dict(states)
