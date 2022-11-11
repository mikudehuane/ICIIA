import math

import pandas as pd
import torch
from tqdm import tqdm

import config
import dataset
import model
import parse
import widgets
import os.path as osp
import logging
import argparse
import sys


def config_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-ds', '--dataset', default='inaturalist',
                        help='dataset name')
    parser.add_argument('--target', default='Smiling', type=str,
                        help='target attribute to predict (for celeba)')
    parser.add_argument("-mn", "--model_name", default='efficientnet-b0', type=str,
                        help="the model to be used (to generate intermediate features")
    parser.add_argument("-tl", "--top_layer", default="multihead_attention",
                        help="top layer to use")

    parser.add_argument('--device', default=None, type=str,
                        help="device to use, if specified, will overwrite the one in config.py")

    parser.add_argument('-nl', '--num_layers', default=2, type=int,
                        help="number of layers to use")
    parser.add_argument('-nh', '--num_heads', default=4, type=int,
                        help="number of heads in multihead attention, also affecting the number of hidden units in fc")
    parser.add_argument('-np', '--num_partitions', default=1, type=int,
                        help="input will first be partitioned and then fed into the encoder,"
                             "if specified as 0, use no affine transformation")
    parser.add_argument('--seed', default=2, type=int,
                        help='seed for all randomers')
    parser.add_argument('--random_batch_size', action='store_true', default=False,
                        help='whether to use random batch size')

    args, unparsed_args = parser.parse_known_args()

    # args check
    if unparsed_args:
        msg = 'Found unrecognized cl arguments: {}'.format(unparsed_args)
        raise RuntimeError(msg)
    return args


def main():
    args = config_args()
    if args.seed is not None:
        config.config_seed(args.seed)
    if args.device is not None:
        config.DEVICE = torch.device(args.device)

    log_dir = widgets.config_log_dir_and_write_args('adapt', args, level=logging.INFO)
    log_dir_name = osp.basename(log_dir)

    if args.dataset == 'celeba' and args.target == 'all':
        dim_output = len(dataset.ALL_CELEBA_ATTRS)  # multi-label classification
    else:
        dim_output = config.NUM_CLASSES[args.dataset]
    encoder_layer_cls = (model.attention.PartitionedTransformerEncoderLayer if args.num_partitions == 1
                         else model.attention.PartitionedTransformerEncoderLayerEfficient)
    predictor = model.attention.PartitionedTransformerEncoder(
        num_layers=args.num_layers, num_heads=args.num_heads, dropout=0.1,
        dim_input=config.FEATURE_DIM[args.model_name], dim_output=dim_output,
        num_partitions=args.num_partitions, encoder_layer_cls=encoder_layer_cls,
        transpose=True
    )
    if args.random_batch_size:
        insert = 'rbs_'
    else:
        insert = ''
    model_name = f'{args.dataset}_{args.model_name}_{insert}{args.num_layers}-{args.num_partitions}p-s{args.seed}.pth'
    model_path = osp.join(config.MODELS_DIR, model_name)
    # load model
    predictor.to(config.DEVICE)
    predictor.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    print(f'Loaded model from {model_path}')

    base_predictor = parse.parse_base_head(args)
    base_predictor.eval()

    if args.dataset == 'inaturalist':
        _, test_dataset = parse.get_head_dataset(args.dataset, args.model_name)
        _, user_test_indices = dataset.inaturalist.get_user_indices()
    else:
        raise NotImplementedError("dataset {} not implemented".format(args.dataset))

    num_test_samples_max = max(len(x) for x in user_test_indices.values())  # 147
    num_test_samples_min = min(len(x) for x in user_test_indices.values())  # 20
    cache_count = list(range(num_test_samples_max))
    result_frame = pd.DataFrame(
        columns=['acc', 'flops'],
        index=cache_count)
    for cache_count in tqdm(cache_count):
        predicts = []  # predictor output for the last sample for each user
        labels = []

        for uid, indices in tqdm(user_test_indices.items()):
            for target_idx in indices:
                remain_idxs = [idx for idx in indices if idx != target_idx]
                valid_idxs = remain_idxs[:cache_count] + [target_idx]  # can be smaller than cache_count if no enough samples
                batch = [test_dataset[idx][0] for idx in valid_idxs]
                label = test_dataset[target_idx][1]
                batch = torch.stack(batch)
                batch = batch.to(config.DEVICE)
                preds = predictor(batch)
                pred = preds[-1]
                pred = torch.argmax(pred)
                labels.append(label)
                predicts.append(pred)

        predicts = torch.stack(predicts).to(config.DEVICE)
        labels = torch.stack(labels).to(config.DEVICE)

        # ISA
        corrects = (predicts == labels)
        acc = corrects.sum().item() / len(corrects)
        print(f'{cache_count} samples, acc: {acc:.4f}')

        # overhead
        seq_input = torch.zeros((cache_count + 1, config.FEATURE_DIM[args.model_name])).to(config.DEVICE)
        seq_flops, _ = model.extended_profile(predictor, (seq_input,))

        result_frame.loc[cache_count, 'acc'] = acc
        result_frame.loc[cache_count, 'flops'] = seq_flops

    result_frame.to_csv(osp.join(log_dir, 'result.csv'))


if __name__ == "__main__":
    main()
