import argparse
import sys

import torch

import config
import widgets
from parse import parse_base_head, parse_metrics, parse_head_dataset


def config_args():
    unparsed_args = sys.argv[1:]
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-mn', '--model_name', type=str, default='efficientnet-b4')
    parser.add_argument('-bs', '--batch_size', type=int, default=1024)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('-ds', '--dataset', type=str, default='imagenet',
                        choices=('imagenet', 'femnist', 'celeba', 'ucf101', 'inaturalist'),
                        help='dataset to extract features')
    parser.add_argument('--split', default='sample', choices=('sample', 'user'),
                        help='dataset splitting method, sample or user')
    parser.add_argument('--soft_reserve_ratio', default=1.0, type=float,
                        help='reserve this ratio of samples in the super class, and others as random')
    parser.add_argument('--target', default='Smiling', type=str,
                        help='target attribute to predict (for celeba)')
    parser.add_argument("--num_workers", default=0, type=int,
                        help="number of workers to load data")
    parser.add_argument("--train_ratio", default=0.8, type=float,
                        help="train-val-split ratio of training set")

    args, unparsed_args = parser.parse_known_args(args=unparsed_args, namespace=None)

    parser_help = argparse.ArgumentParser(parents=[parser], prog="eval_base",
                                          description="evaluate base model")
    parser_help.parse_known_args()

    if unparsed_args:
        msg = 'Found unrecognized cl arguments: {}'.format(unparsed_args)
        raise RuntimeError(msg)

    return args


def main():
    args = config_args()

    if args.device is not None:
        config.DEVICE = torch.device(args.device)

    predictor = parse_base_head(args)

    predictor.to(device=config.DEVICE)
    predictor.eval()

    loader_train, loader_val, loader_test = parse_head_dataset(args)
    metrics_2b_eval = parse_metrics(args)
    if args.dataset == 'celeba' and args.target == 'all':
        metrics_2b_eval['accuracy_per_task'] = widgets.evaluation.AccuracyPerTask

    for dset_name, loader in zip(('val', 'test'), (loader_val, loader_test)):
        metric_vals = widgets.eval_loader(predictor, loader, verbose=True, metrics=metrics_2b_eval)
        msg_builder = []
        for metric_name, metric_value in metric_vals.items():
            msg_builder.append('{}: {}'.format(metric_name, metric_value))
        msg_content = '{dset_name}, {metrics}'.format(dset_name=dset_name, metrics=' '.join(msg_builder))
        print(msg_content)


if __name__ == '__main__':
    main()
