import argparse
import json
import os
import os.path as osp

import numpy as np
from tqdm import tqdm

from constants import DATASETS


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name',
                        help='name of dataset to parse; default: sent140;',
                        type=str,
                        choices=DATASETS,
                        default='sent140')

    args = parser.parse_args()

    parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_fd = os.path.join(parent_path, args.name, 'data')

    with open(osp.join(data_fd, 'clients_split', 'all.json'), 'r') as f:
        full_index = json.load(f)
    num_samples = sum(len(x) for x in full_index.values())

    input_dir = osp.join(data_fd, 'clients_split', 'data')
    input_fnames = os.listdir(input_dir)
    with open(osp.join(input_dir, input_fnames[0])) as f:
        sample = json.load(f)
    num_dims = len(sample)
    dump_obj = np.zeros((num_samples, num_dims), dtype=np.float32)
    for input_fname in tqdm(input_fnames):
        input_path = osp.join(input_dir, input_fname)
        with open(input_path) as f:
            sample = json.load(f)
        index = int(input_fname.split('.')[0])
        dump_obj[index] = sample

    output_path = os.path.join(data_fd, 'clients_split', 'all_data.pkl')
    with open(output_path, 'wb') as f:
        dump_obj.dump(f)


if __name__ == '__main__':
    main()
