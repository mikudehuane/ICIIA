import json
import logging
import os
import sys
import argparse
import os.path as osp
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List

import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

import config
import dataset
import parse
import widgets
from parse import parse_optimizer, parse_base_model, parse_base_head, parse_criterion, parse_metrics

_logger = logging.getLogger(__name__)


def config_args():
    unparsed_args = sys.argv[1:]

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-ds', '--dataset', default='imagenet', choices=('imagenet', 'femnist', 'celeba', 'ucf101'),
                        help='dataset name')
    parser.add_argument('--target', default='Smiling', type=str,
                        help='target attribute to predict (for celeba)')

    parser.add_argument("-ne", "--num_epoches", default=100, type=int,
                        help="max number of training epoches")
    parser.add_argument("-opt", "--optimizer", default="sgd", type=str, choices=("sgd", 'adam', 'RMSProp'),
                        help="optimizer to use")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="L2 regularization coefficient")
    parser.add_argument('-bs', '--batch_size', default=16, type=int,
                        help="batch size")
    parser.add_argument("--train_ratio", default=0.8, type=float,
                        help="train-val-split ratio of training set")
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for all randomers')

    parser.add_argument("--num_workers", default=0, type=int,
                        help="number of workers to load data")
    parser.add_argument('--device', default=None, type=str,
                        help="device to use, if specified, will overwrite the one in config.py")
    parser.add_argument('--early_stop_num_epoches', default=10, type=int,
                        help="early stop if no improvement after this many epoches")

    parser.add_argument("-mn", "--model_name", default='efficientnet-b4', type=str,
                        help="the model to be used (to generate intermediate features")

    args, unparsed_args = parser.parse_known_args(args=unparsed_args, namespace=None)

    parser1 = argparse.ArgumentParser(add_help=False)
    parser1.add_argument("-lr", "--learning_rate", type=float,
                         default=0.01,
                         dest="lr", help="default learning rate of each layer, may be overwritten by other args")
    parser1.add_argument("--lr_decay", default=1.0, type=float,
                         help="learning rate decay (lr *= lr_decay each time)")
    parser1.add_argument('--momentum', default=0.0, type=float,
                         help="momentum parameter for optimizers")
    parser1.add_argument("--decay_every", default=1.0, type=float,
                         help="decay learning rate every this many epochs (can be fractional)")
    if args.dataset == 'imagenet':
        parser1.add_argument('--soft_reserve_ratio', default=1.0, type=float,
                             help='reserve this ratio of samples in the super class, and others as random')

    args, unparsed_args = parser1.parse_known_args(args=unparsed_args, namespace=args)

    parser_help = argparse.ArgumentParser(parents=[parser, parser1], prog="finetune",
                                          description="finetune a linear head for each client")
    parser_help.parse_known_args()

    if unparsed_args:
        msg = 'Found unrecognized cl arguments: {}'.format(unparsed_args)
        raise RuntimeError(msg)

    return args


def main():

    args = config_args()
    config.config_seed(args.seed)
    if args.device is not None:
        config.DEVICE = torch.device(args.device)
    log_dir = widgets.config_log_dir_and_write_args("finetune", args, level=logging.INFO)
    log_dir_name = osp.basename(log_dir)
    info_path = osp.join(log_dir, "info.txt")
    os.makedirs(log_dir, exist_ok=True)

    if args.dataset == 'imagenet':
        train_val_dataset, test_dataset = parse.get_head_dataset(args.dataset, args.model_name)

        user_train_indices, user_val_indices, user_test_indices = dataset.imagenet.get_final_user_split(
            train_val_dataset, test_dataset, train_ratio=args.train_ratio, soft_reserve_ratio=args.soft_reserve_ratio)

        _logger.info("num users: train({}), val({}), test({})".format(
            len(user_train_indices), len(user_val_indices), len(user_test_indices)))

        # get_dataset_* is invoked in the user loop and will use the variables in the loop
        # the function depends on the dataset, and thus defined here
        def get_dataset_train():
            return dataset.RemappedDataset(train_val_dataset, train_index)

        def get_dataset_val():
            return dataset.RemappedDataset(train_val_dataset, val_index)

        def get_dataset_test():
            return dataset.RemappedDataset(test_dataset, test_index)
    elif args.dataset == 'femnist':
        user_train_indices, user_val_indices, user_test_indices = dataset.get_femnist_user_indices(
            osp.join(config.DATA_FEMNIST_DIR, 'clients'), train_ratio=args.train_ratio)

        full_dataset = parse.get_head_dataset(args.dataset, args.model_name)

        def get_dataset_train():
            return dataset.RemappedDataset(full_dataset, train_index)

        def get_dataset_val():
            return dataset.RemappedDataset(full_dataset, val_index)

        def get_dataset_test():
            return dataset.RemappedDataset(full_dataset, test_index)
    elif args.dataset == 'celeba':
        user_train_indices, user_val_indices, user_test_indices = dataset.get_celeba_user_indices(
            train_ratio=args.train_ratio)
        train_val_dataset, test_dataset = parse.get_head_dataset(args.dataset, args.model_name, target=args.target)

        def get_dataset_train():
            return dataset.RemappedDataset(train_val_dataset, train_index)

        def get_dataset_val():
            return dataset.RemappedDataset(train_val_dataset, val_index)

        def get_dataset_test():
            return dataset.RemappedDataset(test_dataset, test_index)
    elif args.dataset == 'ucf101':
        train_val_dataset, test_dataset = parse.get_head_dataset(args.dataset, args.model_name)
        train_val_dataset_base = dataset.ucf101.UCF101Dataset(
            lambda x: np.zeros(()), video_index=config.DATA_UCF101_TRAIN_INDEX_PATH)
        test_dataset_base = dataset.ucf101.UCF101Dataset(
            lambda x: np.zeros(()), video_index=config.DATA_UCF101_TEST_INDEX_PATH)
        # check for consistency
        assert [x[1] for x in train_val_dataset] == [x[1] for x in train_val_dataset_base]
        assert [x[1] for x in test_dataset] == [x[1] for x in test_dataset_base]
        user_train_indices, user_val_indices, user_test_indices = dataset.ucf101.get_final_user_split(
            train_val_dataset_base, test_dataset_base, train_ratio=args.train_ratio)

        _logger.info("num users: train({}), val({}), test({})".format(
            len(user_train_indices), len(user_val_indices), len(user_test_indices)))

        # get_dataset_* is invoked in the user loop and will use the variables in the loop
        # the function depends on the dataset, and thus defined here
        def get_dataset_train():
            return dataset.RemappedDataset(train_val_dataset, train_index)

        def get_dataset_val():
            return dataset.RemappedDataset(train_val_dataset, val_index)

        def get_dataset_test():
            return dataset.RemappedDataset(test_dataset, test_index)
    else:
        raise NotImplementedError("dataset {} not implemented".format(args.dataset))
    orig_predictor = parse_base_head(args)
    orig_predictor.to('cpu')

    assert list(user_train_indices.keys()) == list(user_test_indices.keys()), "train-val-test split not consistent"
    assert list(user_train_indices.keys()) == list(user_val_indices.keys()), "train-val-test split not consistent"

    with open(info_path, 'a') as info_f:
        sample_model = nn.Linear(config.FEATURE_DIM[args.model_name], config.NUM_CLASSES[args.dataset], device='cpu')
        num_params_per_model = sum(p.numel() for p in sample_model.parameters())
        num_flops_per_model = config.FEATURE_DIM[args.model_name] * config.NUM_CLASSES[args.dataset]

        info_f.write(
            'FLOPs: {}, Params: {}\n'.format(num_flops_per_model,
                                             num_params_per_model * len(user_test_indices))
        )

        del sample_model

    # evaluate per user
    all_users_meta = dict()
    with open(osp.join(log_dir, 'user_result.csv'), 'w') as user_result_f:
        user_result_f.write(
            'user,val_acc,val_loss,val_num_samples,test_acc,test_loss,test_num_samples,num_classes,stop_epoch\n')
        for user_id in tqdm(user_train_indices, log_dir_name):
            logging.info("examine user id: {}".format(user_id))

            train_index = user_train_indices[user_id]
            val_index = user_val_indices[user_id]
            test_index = user_test_indices[user_id]

            dataset_train = get_dataset_train()
            dataset_val = get_dataset_val()
            dataset_test = get_dataset_test()

            loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            loader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

            predictor = deepcopy(orig_predictor)
            predictor.to(config.DEVICE)
            optimizer = parse_optimizer(args, predictor)
            criterion = parse_criterion(args)
            lr_scheduler = ExponentialLR(optimizer, gamma=args.lr_decay)
            metrics_2b_eval = parse_metrics(args)

            global_it = 0
            best_metric = -1.
            epoch_num = 0
            is_early_stopped = False
            decay_every_num_iterations = round(args.decay_every * len(loader_train))
            """
            {
                'testset': {'cre_loss': float, 'accuracy': float, ...},
                'valset': {'cre_loss': float, 'accuracy': float, ...}
            }
            """
            best_meta: Dict = dict()
            best_meta['epoch'] = -1
            assert isinstance(decay_every_num_iterations, int)

            def flush_all():
                user_result_f.flush()
                sys.stdout.flush()
                config.logging_stream.flush()

            def log_results(_msg, _metric_vals, _set_name):
                _msg_builder = []
                for _metric_name, _metric_value in _metric_vals.items():
                    _msg_builder.append('{}: {}'.format(_metric_name, _metric_value))
                _msg_content = _msg.format(metrics=' '.join(_msg_builder), epoch=epoch_num, user_id=user_id)
                _logger.info(_msg_content)
                # flush_all()

            def eval_all_and_save():
                nonlocal best_meta

                prev_epoch = best_meta['epoch']
                current_meta = dict()
                # evaluate testset
                _metric_vals_test = widgets.eval_loader(predictor, loader_test, verbose=False, metrics=metrics_2b_eval)
                log_results('user {user_id}, epoch {epoch} testset, {metrics}', _metric_vals_test, 'test')
                # evaluate validation set
                _metric_vals_val = widgets.eval_loader(predictor, loader_val, verbose=False, metrics=metrics_2b_eval)
                log_results('user {user_id}, epoch {epoch} valset, {metrics}', _metric_vals_val, 'val')

                current_meta['testset'] = deepcopy(_metric_vals_test)
                current_meta['valset'] = deepcopy(_metric_vals_val)
                current_meta['epoch'] = epoch_num

                _metric_val = _metric_vals_val['accuracy']
                nonlocal best_metric
                if _metric_val > best_metric:
                    best_meta = deepcopy(current_meta)
                    best_metric = _metric_val
                else:
                    if epoch_num - prev_epoch >= args.early_stop_num_epoches:
                        _logger.info('user {}, early stop at epoch {}'.format(user_id, epoch_num))
                        return False
                return True

            for epoch_num in range(args.num_epoches):
                if not eval_all_and_save():
                    is_early_stopped = True
                    break

                for epoch_it, (inputs, labels) in enumerate(loader_train):
                    inputs = inputs.to(config.DEVICE)
                    labels = labels.to(config.DEVICE)

                    optimizer.zero_grad()
                    predictor.train()
                    outputs = predictor(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    nn.utils.clip_grad_norm_(predictor.parameters(), 5.0)
                    optimizer.step()

                    global_it += 1
                    if global_it % decay_every_num_iterations == 0:
                        lr_scheduler.step()

            if not is_early_stopped:
                epoch_num += 1  # final epoch
                is_early_stopped = not eval_all_and_save()
                if not is_early_stopped:
                    epoch_num = 'not-early-stopped'
            loss_key = 'bce_loss' if args.dataset == 'celeba' and args.target == 'all' else 'cre_loss'
            user_result_f.write('{},{},{},{},{},{},{},{},{}\n'.format(
                user_id,
                best_meta['valset']['accuracy'], best_meta['valset'][loss_key], best_meta['valset']['num_samples'],
                best_meta['testset']['accuracy'], best_meta['testset'][loss_key], best_meta['testset']['num_samples'],
                best_meta['testset'].get('num_classes', 0),
                epoch_num)
            )
            flush_all()
            all_users_meta[user_id] = deepcopy(best_meta)

        def get_full_metric(subset_name, metric_name):
            total = np.sum([
                meta[subset_name][metric_name] * meta[subset_name]['num_samples']
                for meta in all_users_meta.values()
            ])
            div = np.sum([meta[subset_name]['num_samples'] for meta in all_users_meta.values()])
            return total / div

        with open(info_path, 'a') as info_f:
            metrics = {
                dname: {
                    mname: get_full_metric(dname, mname)
                    for mname in (loss_key, 'accuracy')
                }
                for dname in ('testset', 'valset')
            }
            info_f.write(json.dumps(metrics) + '\n')


if __name__ == '__main__':
    main()
