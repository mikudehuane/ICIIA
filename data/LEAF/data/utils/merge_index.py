import argparse
import json
import os
import os.path as osp

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

    input_dir = osp.join(data_fd, 'clients_split', 'clients')
    input_fnames = os.listdir(input_dir)
    input_fnames.sort()  # avoid randomness
    input_paths = [osp.join(input_dir, fname) for fname in input_fnames]

    output = {'train': [], 'test': []}
    for input_path in input_paths:
        with open(input_path, 'r') as f:
            data = json.load(f)
            output['train'].extend(data['train'])
            output['test'].extend(data['test'])

    output_path = os.path.join(data_fd, 'clients_split', 'all.json')
    with open(output_path, 'w') as f:
        json.dump(output, f)


if __name__ == '__main__':
    main()
