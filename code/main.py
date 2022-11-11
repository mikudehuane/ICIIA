from __future__ import division

import argparse
import json
import logging
import os
import sys
from copy import deepcopy
from typing import Dict

import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import model
import dataset
import os.path as osp
import config
import widgets
from model.attention import TransformerEncoderWrapper
from parse import parse_dataset, parse_optimizer, parse_predictor, parse_criterion, parse_metrics, parse_base_model
from widgets import eval_loader, dump_json


_logger = logging.getLogger(__name__)


def config_args():
    unparsed_args = sys.argv[1:]

    parser = argparse.ArgumentParser(add_help=False)

    # main arguments
    parser.add_argument('--phase', default='head_per', choices=('head_per', 'pretrain'),
                        type=str, help='phase to run')
    parser.add_argument('-ds', '--dataset', default='imagenet',
                        choices=('imagenet', 'femnist', 'celeba', 'ucf101', 'inaturalist'),
                        help='dataset name')
    parser.add_argument('--target', default='Smiling', type=str,
                        help='target attribute to predict (for celeba)')
    parser.add_argument("-mn", "--model_name", default='efficientnet-b4', type=str,
                        help="the model to be used (to generate intermediate features")
    parser.add_argument("-tl", "--top_layer", default="multihead_attention",
                        choices=("multihead_attention", "ff", 'prompt'),
                        help="top layer to use")
    parser.add_argument('--split', default='sample', choices=('sample', 'user'),
                        help='dataset splitting method, sample or user')

    # optimization arguments
    parser.add_argument("-ne", "--num_epoches", default=100, type=int,
                        help="number of training epoches")
    parser.add_argument("-opt", "--optimizer", default="sgd", type=str, choices=("sgd", 'adam', 'RMSProp'),
                        help="optimizer to use")
    parser.add_argument('-bs', '--batch_size', default=16, type=int,
                        help="batch size, if 0, each user as a batch")
    parser.add_argument('--random_batch_size', action='store_true', default=False,
                        help='whether to use random batch size')
    parser.add_argument("--train_ratio", default=0.8, type=float,
                        help="train-val-split ratio of training set")
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for all randomers')

    parser.add_argument('--eval_mode', default='batch', choices=('batch', 'stored'),
                        help='evaluation mode, batch: evaluate directly on each batch, '
                             'stored: evaluate with the training data')
    parser.add_argument("--print_every", default=100, type=int,
                        help="print the running results every this many iterations")
    parser.add_argument("--num_workers", default=0, type=int,
                        help="number of workers to load data")
    parser.add_argument('--device', default=None, type=str,
                        help="device to use, if specified, will overwrite the one in config.py")
    parser.add_argument('--early_stop_num_epoches', default=10, type=int,
                        help="early stop if no improvement after this many epoches")

    # top layer arguments
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="dropout rate")
    parser.add_argument('-nl', '--num_layers', default=1, type=int,
                        help="number of layers to use")
    parser.add_argument('-nh', '--num_heads', default=1, type=int,
                        help="number of heads in multihead attention, also affecting the number of hidden units in fc")
    parser.add_argument('-np', '--num_partitions', default=1, type=int,
                        help="input will first be partitioned and then fed into the encoder,"
                             "if specified as 0, use no affine transformation")
    parser.add_argument('--no_transpose', action='store_false', dest='transpose',
                        help='whether to transpose the input after diag_mm')

    args, unparsed_args = parser.parse_known_args(args=unparsed_args, namespace=None)

    parser1 = argparse.ArgumentParser(add_help=False)
    parser1.add_argument("-vbs", "--val_batch_size", default=args.batch_size, type=int,
                         help="batch size while validating")
    parser1.add_argument("-tebs", "--test_batch_size", default=args.batch_size, type=int,
                         help="batch size while testing")
    weight_decay_default = 0.0 if args.phase == 'head_per' else 5e-4
    parser1.add_argument("--weight_decay", default=weight_decay_default, type=float,
                        help="L2 regularization coefficient")
    parser1.add_argument("-lr", "--learning_rate", type=float,
                         default=0.01,
                         dest="lr", help="default learning rate of each layer, may be overwritten by other args")
    parser1.add_argument("--lr_decay", default=1.0, type=float,
                         help="learning rate decay (lr *= lr_decay each time)")
    parser1.add_argument('--momentum', default=0.0, type=float,
                         help="momentum parameter for optimizers")
    parser1.add_argument("--decay_every", default=1.0, type=float,
                         help="decay learning rate every this many epochs (can be fractional)")
    parser1.add_argument('-nbs', '--num_batches_per_step', default=1, type=int,
                         help='number of batches to accumulate gradients before updating parameters')
    parser1.add_argument('--dim_prompt', type=int, help='dimension of user embedding in prompt tuning')
    parser1.add_argument('--soft_reserve_ratio', default=1.0, type=float,
                         help='reserve this ratio of samples in the super class, and others as random')

    args, unparsed_args = parser1.parse_known_args(args=unparsed_args, namespace=args)

    parser_help = argparse.ArgumentParser(parents=[parser, parser1], prog="main",
                                          description="main script")
    parser_help.parse_known_args()

    # args check
    if unparsed_args:
        msg = 'Found unrecognized cl arguments: {}'.format(unparsed_args)
        raise RuntimeError(msg)
    if args.eval_mode == 'stored':
        assert args.phase == 'head_per', 'eval_mode stored only works for head_per phase'
        assert args.dataset == 'celeba', 'only celeba dataset supports stored evaluation mode'
        assert args.batch_size == 0, 'batch_size must be 0 in stored evaluation mode'

    return args


class ZipLoader(object):
    def __init__(self, loader_kernel, loader_prompt):
        self.loader_kernel = loader_kernel
        self.loader_prompt = loader_prompt
        assert len(loader_kernel) == len(loader_prompt), 'kernel and prompt loaders must have the same length'

    def __iter__(self):
        for (x, y), (x_p, y_p) in zip(self.loader_kernel, self.loader_prompt):
            yield x, y, x_p

    def __len__(self):
        return len(self.loader_kernel)


def main():

    def flush_all():
        tb_writer.flush()
        sys.stdout.flush()
        config.logging_stream.flush()

    def log_results(_msg, _metric_vals, _set_name):
        _msg_builder = []
        for _metric_name, _metric_value in _metric_vals.items():
            if not isinstance(_metric_value, list):
                tb_writer.add_scalar('{}_{}'.format(_set_name, _metric_name), _metric_value, epoch_num)
            _msg_builder.append('{}: {}'.format(_metric_name, _metric_value))
        _msg_content = _msg.format(metrics=' '.join(_msg_builder), epoch=epoch_num)
        _logger.info(_msg_content)
        flush_all()

    def eval_all_and_save():
        nonlocal best_meta

        prev_epoch = best_meta['epoch']
        current_meta = dict()
        if args.eval_mode == 'batch':
            # evaluate testset
            _metric_vals_test = eval_loader(predictor, loader_test, verbose=True, metrics=metrics_2b_eval)
            # evaluate validation set
            _metric_vals_val = eval_loader(predictor, loader_val, verbose=True, metrics=metrics_2b_eval)
        elif args.eval_mode == 'stored':
            wrapped_predictor = TransformerEncoderWrapper(predictor)
            _metric_vals_test = eval_loader(
                wrapped_predictor, ZipLoader(loader_test, loader_train), verbose=True, metrics=metrics_2b_eval)
            _metric_vals_val = eval_loader(
                wrapped_predictor, ZipLoader(loader_val, loader_train), verbose=True, metrics=metrics_2b_eval)
        else:
            raise ValueError('unknown eval_mode: {}'.format(args.eval_mode))
        log_results('epoch {epoch} testset, {metrics}', _metric_vals_test, 'test')
        log_results('epoch {epoch} valset, {metrics}', _metric_vals_val, 'val')

        current_meta['testset'] = deepcopy(_metric_vals_test)
        current_meta['valset'] = deepcopy(_metric_vals_val)
        current_meta['epoch'] = epoch_num

        _model_name = 'running_e{:0>4d}'.format(epoch_num)
        # torch.save(predictor.state_dict(), osp.join(model_dir, '{}.pth'.format(_model_name)))
        dump_json(current_meta, osp.join(model_dir, '{}.json'.format(_model_name)))

        _metric_val = _metric_vals_val['accuracy']
        nonlocal best_metric
        if _metric_val > best_metric:
            best_meta = deepcopy(current_meta)
            best_metric = _metric_val
            torch.save(predictor.state_dict(), osp.join(model_dir, 'best.pth'))
            with open(osp.join(model_dir, 'best.json'), 'w') as dump_fout:
                for key, val in best_meta.items():
                    dump_fout.write('{}: {}\n'.format(key, json.dumps(val)))
        else:  # if not improved, judge whether to stop
            if epoch_num - prev_epoch >= args.early_stop_num_epoches:
                _logger.info('early stop at epoch {}'.format(epoch_num))
                with open(info_path, 'a') as early_stop_f:
                    early_stop_f.write('stop (before) epoch: {}\n'.format(epoch_num))
                return False
        return True

    best_meta: Dict = dict()
    best_meta['epoch'] = -1

    args = config_args()
    if args.seed is not None:
        config.config_seed(args.seed)
    if args.device is not None:
        config.DEVICE = torch.device(args.device)
    # logging config
    log_dir = widgets.config_log_dir_and_write_args(args.phase, args, level=logging.INFO)
    log_dir_name = osp.basename(log_dir)

    model_dir = osp.join(log_dir, 'models')  # dir to save the models, together with a json file recording the metrics
    tensorboard_dir = log_dir
    info_path = osp.join(log_dir, 'info.txt')  # saving result msgs
    os.makedirs(model_dir, exist_ok=False)
    tb_writer = SummaryWriter(tensorboard_dir)

    loader_train, loader_val, loader_test = parse_dataset(args)

    if args.phase == 'head_per':
        assert isinstance(loader_test.batch_sampler, dataset.UserBatchSampler), 'test set must be user batch sampler'
        predictor = parse_predictor(args, num_prompts=len(loader_test.batch_sampler.user_indices))
    elif args.phase == 'pretrain':
        predictor = parse_base_model(args, load_pretrained=False)
    else:
        raise ValueError('unknown phase: {}'.format(args.phase))
    _logger.info('network structure:\n' + str(predictor))

    optimizer = parse_optimizer(args, predictor)
    criterion = parse_criterion(args)
    lr_scheduler = ExponentialLR(optimizer, gamma=args.lr_decay, verbose=True)
    metrics_2b_eval = parse_metrics(args)
    accuracy_metric_func = metrics_2b_eval['accuracy']
    accuracy_metric = accuracy_metric_func()

    global_it = 0
    best_metric = -1.
    epoch_num = 0
    is_early_stopped = False
    decay_every_num_iterations = round(args.decay_every * len(loader_train))
    assert isinstance(decay_every_num_iterations, int)
    is_first_batch = True

    for epoch_num in range(args.num_epoches):
        if not eval_all_and_save():
            is_early_stopped = True
            break
        # noinspection PyUnresolvedReferences
        _logger.info('Epoch {}/{} start'.format(epoch_num + 1, args.num_epoches))

        running_loss = 0.0
        total_cnt = 0

        optimizer.zero_grad()
        for epoch_it, sample in enumerate(tqdm(
                loader_train, '{} epoch [{}/{}], training'.format(log_dir_name, epoch_num, args.num_epoches))):
            # parse sample to inputs, label, inputs will be a list of inputs
            if args.top_layer == 'prompt':
                assert len(sample) == 3, 'prompt requires 3 inputs'
                inputs, labels, prompt = sample
                assert prompt.device == torch.device('cpu')
                inputs = [inputs, prompt]
            else:
                assert len(sample) == 2, 'len(sample) = {}'.format(len(sample))
                inputs, labels = sample
                inputs = [inputs,]
            inputs[0] = inputs[0].to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            if is_first_batch:
                is_first_batch = False
                num_flops, num_params = model.extended_profile(predictor, inputs, report_missing=True)
                with open(info_path, 'a') as f:
                    f.write('FLOPs: {}, Params: {}\n'.format(num_flops / inputs[0].size(0), num_params))

            predictor.train()
            outputs = predictor(*inputs)
            accuracy_metric.accumulate(outputs, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            if (epoch_it + 1) % args.num_batches_per_step == 0:
                nn.utils.clip_grad_norm_(predictor.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * inputs[0].size(0)
            total_cnt += labels.size(0)

            # print the current loss
            global_it += 1
            if global_it % args.print_every == 0:
                loss_running = running_loss / total_cnt
                acc_running = accuracy_metric.get_result()
                accuracy_metric = accuracy_metric_func()
                tb_writer.add_scalar('running_loss', loss_running, global_it)
                tb_writer.add_scalar('running_acc', acc_running, global_it)
                _logger.info('epoch {} [{}/{}], running loss: {}, accuracy: {}'.format(
                    epoch_num, epoch_it, len(loader_train), loss_running, acc_running))

                running_loss = 0.0
                total_cnt = 0
                # flush_all()
            if global_it % decay_every_num_iterations == 0:
                lr_scheduler.step()
    if not is_early_stopped:
        epoch_num += 1  # final epoch
        is_early_stopped = not eval_all_and_save()
        if not is_early_stopped:
            with open(info_path, 'a') as f:
                f.write('not early stopped!\n')


if __name__ == '__main__':
    main()