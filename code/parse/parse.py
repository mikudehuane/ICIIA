import argparse
import sys
from collections import OrderedDict
from typing import Tuple, Optional, Union
import os.path as osp

import numpy as np
import torch.nn
import torchvision
from efficientnet_pytorch import EfficientNet
from torch import optim, nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import config

import dataset
import model
import logging

import widgets
from model import FemnistCNN, CelebaCNN, C3D
from .utils import get_dataset_name, get_head_dataset, get_base_dataset

_logger = logging.getLogger(__name__)


def parse_optimizer(args: argparse.Namespace, predictor: nn.Module) -> optim.Optimizer:
    if 'num_batches_per_step' in args.__dict__:
        # Logically, the loss is the mean, but accumulated gradient is summed. Thus, the lr should be divided.
        lr = args.lr / args.num_batches_per_step
    else:
        lr = args.lr

    params = None
    if 'phase' in args.__dict__ and args.phase == 'pretrain':
        if args.model_name == 'c3d':
            # different lr for base and top params
            params = [{'params': predictor.get_base_params(), 'lr': lr / 10},
                      {'params': predictor.get_top_params(), 'lr': lr}]
        elif args.model_name.startswith('efficientnet-') and args.dataset == 'inaturalist':
            predictor: EfficientNet
            predictor.parameters()
            params = [{'params': model.get_efficientnet_base_params(predictor), 'lr': lr / 10},
                      {'params': model.get_efficientnet_top_params(predictor), 'lr': lr}]
    if params is None:
        params = predictor.parameters()

    if args.optimizer == 'adam':
        optimizer = optim.Adam(params, weight_decay=args.weight_decay, lr=lr)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(params, weight_decay=args.weight_decay, lr=lr,
                              momentum=args.__dict__.get('momentum', 0))
    elif args.optimizer == 'RMSProp':
        optimizer = optim.RMSprop(params, weight_decay=args.weight_decay, lr=lr,
                                  alpha=0.9, momentum=args.momentum)
    else:
        raise RuntimeError("unrecognized optimizer: {}".format(args.optimizer))

    return optimizer


def parse_dataset(args: argparse.Namespace):
    if args.phase == 'head_per':
        return parse_head_dataset(args)
    elif args.phase == 'pretrain':
        return parse_base_dataset(args)
    else:
        raise AssertionError("phase must be 'head_per' or 'pretrain'")


def parse_base_dataset(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """parse the original dataset to dataloaders

    return train-val-test dataloaders

    used by pretrain.py (not by prep_feats.py since code reason)
    """
    # dataset branch, produce dataset_train, dataset_val, dataset_test
    # these dataset should be indexed continuously from 0,
    # i.e., dataset_train[i] should be valid for all i from 0 to len(dataset_train) - 1
    if args.model_name.startswith('linear_efficientnet-'):
        # use head dataset for linear
        dataset_train_val, dataset_test = get_head_dataset(args.dataset, args.model_name, target=args.target)
        train_indices, val_indices = dataset.train_val_split(list(range(len(dataset_train_val))),
                                                             train_ratio=args.train_ratio, shuffle=True)
        dataset_train = Subset(dataset_train_val, train_indices)
        dataset_val = Subset(dataset_train_val, val_indices)
    else:
        dataset_train, dataset_val, dataset_test = get_base_dataset(
            args.dataset, model_name=args.model_name, train_ratio=args.train_ratio, target=args.target,
        )

    # for pretrain.py, batch size is specified separately, but for prep_feats.py, is specified uniformly
    loader_train = DataLoader(
        dataset_train, shuffle=True, num_workers=args.num_workers,
        batch_size=args.train_batch_size if 'train_batch_size' in args.__dict__ else args.batch_size,
    )
    loader_test = DataLoader(
        dataset_test, shuffle=False, num_workers=args.num_workers,
        batch_size=args.test_batch_size if 'test_batch_size' in args.__dict__ else args.batch_size,
    )
    loader_val = DataLoader(
        dataset_val, shuffle=False, num_workers=args.num_workers,
        batch_size=args.val_batch_size if 'val_batch_size' in args.__dict__ else args.batch_size,
    )
    return loader_train, loader_val, loader_test


def parse_head_dataset(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """return train-val-test dataloaders"""
    train_sampler_kwargs = {'random_batch_size': args.random_batch_size}
    if args.dataset in ('imagenet', 'ucf101'):
        assert args.split == 'sample', "only sample split is supported for imagenet and ucf101"
        # load dataset
        train_val_dataset, test_dataset = get_head_dataset(args.dataset, args.model_name)
        # load split
        if args.dataset == 'imagenet':
            user_train_indices, user_val_indices, user_test_indices = dataset.imagenet.get_final_user_split(
                train_val_dataset, test_dataset, train_ratio=args.train_ratio,
                soft_reserve_ratio=args.soft_reserve_ratio)
        else:
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

        # deal with prompt tuning
        if 'top_layer' in args.__dict__ and args.top_layer == 'prompt':
            # merge train val indices
            assert user_train_indices.keys() == user_val_indices.keys(), "train and val indices must have same keys"
            user_train_val_indices = OrderedDict()
            for key in user_train_indices:
                user_train_val_indices[key] = user_train_indices[key] + user_val_indices[key]
            # create the dataset with prompt (uid)
            train_val_dataset = dataset.DatasetWithUser(train_val_dataset, user_train_val_indices)
            test_dataset = dataset.DatasetWithUser(test_dataset, user_test_indices)

        train_sampler = dataset.UserBatchSampler(user_train_indices, shuffle=True, batch_size=args.batch_size,
                                                 **train_sampler_kwargs)
        val_sampler = dataset.UserBatchSampler(user_val_indices, shuffle=False, batch_size=args.batch_size)
        test_sampler = dataset.UserBatchSampler(user_test_indices, shuffle=False, batch_size=args.batch_size)
        # loader 创建
        loader_train = DataLoader(train_val_dataset, batch_sampler=train_sampler, num_workers=args.num_workers)
        loader_val = DataLoader(train_val_dataset, batch_sampler=val_sampler, num_workers=args.num_workers)
        loader_test = DataLoader(test_dataset, batch_sampler=test_sampler, num_workers=args.num_workers)
    elif args.dataset == 'femnist':
        assert args.split == 'sample', "only sample split is supported for femnist"
        # load dataset
        full_dataset = get_head_dataset(args.dataset, args.model_name)
        # load split
        user_train_indices, user_val_indices, user_test_indices = dataset.get_femnist_user_indices(
            osp.join(config.DATA_FEMNIST_DIR, 'clients'), train_ratio=args.train_ratio)

        train_sampler = dataset.UserBatchSampler(user_train_indices, shuffle=True, batch_size=args.batch_size,
                                                 **train_sampler_kwargs)
        val_sampler = dataset.UserBatchSampler(user_val_indices, shuffle=False, batch_size=args.batch_size)
        test_sampler = dataset.UserBatchSampler(user_test_indices, shuffle=False, batch_size=args.batch_size)

        # deal with prompt tuning
        if 'top_layer' in args.__dict__ and args.top_layer == 'prompt':
            # 合并 train val indices
            assert user_train_indices.keys() == user_val_indices.keys(), "train and val indices must have same keys"
            assert user_train_indices.keys() == user_test_indices.keys(), "train and test indices must have same keys"
            user_full_indices = OrderedDict()
            for key in user_train_indices:
                user_full_indices[key] = user_train_indices[key] + user_val_indices[key] + user_test_indices[key]
            full_dataset = dataset.DatasetWithUser(full_dataset, user_full_indices)

        loader_train = DataLoader(full_dataset, batch_sampler=train_sampler, num_workers=args.num_workers)
        loader_val = DataLoader(full_dataset, batch_sampler=val_sampler, num_workers=args.num_workers)
        loader_test = DataLoader(full_dataset, batch_sampler=test_sampler, num_workers=args.num_workers)
    elif args.dataset == 'celeba':
        user_train_indices, user_val_indices, user_test_indices = dataset.get_celeba_user_indices(
            train_ratio=args.train_ratio, split_method=args.split)

        train_sampler = dataset.UserBatchSampler(user_train_indices, shuffle=True, batch_size=args.batch_size,
                                                 **train_sampler_kwargs)
        val_sampler = dataset.UserBatchSampler(user_val_indices, shuffle=False, batch_size=args.batch_size)
        test_sampler = dataset.UserBatchSampler(user_test_indices, shuffle=False, batch_size=args.batch_size)
        if 'eval_mode' in args.__dict__ and args.eval_mode == 'stored':
            assert args.split == 'sample', "only sample split is supported for eval_mode==stored"
            # ensure same train-val-test order
            train_sampler.register_set_user_order_func(dataset.set_user_order_randomly_func(args.seed))
            val_sampler.register_set_user_order_func(dataset.set_user_order_randomly_func(args.seed))
            test_sampler.register_set_user_order_func(dataset.set_user_order_randomly_func(args.seed))
            assert len(train_sampler) == len(val_sampler) == len(test_sampler), "train/val/test must have same size"

        # load dataset
        train_val_dataset, test_dataset = get_head_dataset(args.dataset, args.model_name, target=args.target)

        # deal with prompt tuning
        if 'top_layer' in args.__dict__ and args.top_layer == 'prompt':
            assert args.split == 'sample', "only sample split is supported for prompt tuning"
            # merge train val indices
            assert user_train_indices.keys() == user_val_indices.keys(), "train and val indices must have same keys"
            user_train_val_indices = OrderedDict()
            for key in user_train_indices:
                user_train_val_indices[key] = user_train_indices[key] + user_val_indices[key]
            # create the dataset with prompt (uid)
            train_val_dataset = dataset.DatasetWithUser(train_val_dataset, user_train_val_indices)
            test_dataset = dataset.DatasetWithUser(test_dataset, user_test_indices)

        loader_train = DataLoader(train_val_dataset, batch_sampler=train_sampler, num_workers=args.num_workers)
        loader_val = DataLoader(train_val_dataset, batch_sampler=val_sampler, num_workers=args.num_workers)
        loader_test = DataLoader(test_dataset, batch_sampler=test_sampler, num_workers=args.num_workers)
    elif args.dataset == 'inaturalist':
        assert args.split == 'user', "only user split is supported for inaturalist"
        user_train_indices, user_val_indices, user_test_indices = dataset.inaturalist.get_user_indices(
            train_ratio=args.train_ratio)

        train_sampler = dataset.UserBatchSampler(user_train_indices, shuffle=True, batch_size=args.batch_size,
                                                 **train_sampler_kwargs)
        val_sampler = dataset.UserBatchSampler(user_val_indices, shuffle=False, batch_size=args.batch_size)
        test_sampler = dataset.UserBatchSampler(user_test_indices, shuffle=False, batch_size=args.batch_size)
        if 'eval_mode' in args.__dict__:
            assert args.eval_mode == 'batch', "only batch eval mode is supported for inaturalist"

        # load dataset
        train_val_dataset, test_dataset = get_head_dataset(args.dataset, args.model_name)

        loader_train = DataLoader(train_val_dataset, batch_sampler=train_sampler, num_workers=args.num_workers)
        loader_val = DataLoader(train_val_dataset, batch_sampler=val_sampler, num_workers=args.num_workers)
        loader_test = DataLoader(test_dataset, batch_sampler=test_sampler, num_workers=args.num_workers)
    else:
        raise RuntimeError("unrecognized dataset: {}".format(args.dataset))

    return loader_train, loader_val, loader_test


def parse_predictor(args, *, num_prompts: Optional[int] = None):
    """parse the predictor fed with the hidden features"""
    if args.dataset == 'celeba' and args.target == 'all':
        dim_output = len(dataset.ALL_CELEBA_ATTRS)  # multi-label classification
    else:
        dim_output = config.NUM_CLASSES[args.dataset]

    if args.top_layer == 'multihead_attention':
        encoder_layer_cls = (model.attention.PartitionedTransformerEncoderLayer if args.num_partitions == 1
                             else model.attention.PartitionedTransformerEncoderLayerEfficient)
        predictor = model.attention.PartitionedTransformerEncoder(
            num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
            dim_input=config.FEATURE_DIM[args.model_name], dim_output=dim_output,
            num_partitions=args.num_partitions, encoder_layer_cls=encoder_layer_cls,
            transpose=args.transpose
        )
    elif args.top_layer == 'ff':
        predictor = model.attention.VanillaFF(
            embed_dim=config.FEATURE_DIM[args.model_name], num_layers=args.num_layers,
            dim_output=dim_output, dropout=args.dropout,
            partitions=args.num_partitions
        )
    elif args.top_layer == 'prompt':
        assert isinstance(num_prompts, int), "num_prompts must be specified for prompt tuning"
        predictor = model.PromptFF(dim_input=config.FEATURE_DIM[args.model_name],
                                   dim_prompt=args.dim_prompt,
                                   dim_output=dim_output,
                                   num_prompts=num_prompts, emb_cpu=False)
        base_head = parse_base_head(args)
        predictor.weights.fc.load_state_dict(base_head.state_dict())
    else:
        raise AssertionError("top_layer must be 'multihead_attention' or 'ff'")

    predictor.to(device=config.DEVICE)
    return predictor


def parse_base_model(args, load_pretrained=True):
    """parse the base predicting model

    Args:
        args: command-line arguments
        load_pretrained: whether to load pretrained weights (by pretrain.py)
            - for some models, the pretrained weights from pytorch are always loaded
    """
    dataset_name = get_dataset_name(args.dataset, target=args.target)
    if args.dataset == 'imagenet':
        assert load_pretrained, "imagenet must load pretrained weights"

    if args.model_name.startswith('efficientnet-'):
        predictor = EfficientNet.from_pretrained(args.model_name)
        if args.dataset == 'celeba':  # modify head for celeba
            if args.target == 'all':
                predictor._fc = nn.Linear(predictor._fc.in_features, len(dataset.ALL_CELEBA_ATTRS))  # multihead linear, each for one attribute
            else:
                predictor._fc = nn.Linear(predictor._fc.in_features, config.NUM_CLASSES[args.dataset])

            if load_pretrained:
                path = osp.join(config.MODELS_DIR, '{}_{}.pth'.format(dataset_name, args.model_name))
                if osp.exists(path):
                    widgets.load_states(predictor, path)
                else:
                    # prep_feats.py use the un pretrained efficientnet-b7 directly
                    assert sys.argv[0] == 'prep_feats.py' and args.model_name == 'efficientnet-b7'
                    _logger.warning('the specified path {} does not exist, skip loading'.format(path))
        elif args.dataset == 'inaturalist':
            predictor._fc = nn.Linear(predictor._fc.in_features, config.NUM_CLASSES[args.dataset])
            if load_pretrained:
                path = osp.join(config.MODELS_DIR, '{}_{}.pth'.format(dataset_name, args.model_name))
                widgets.load_states(predictor, path)
        elif args.dataset == 'imagenet':
            pass
        else:
            raise RuntimeError("unrecognized dataset: {}".format(args.dataset))
    elif args.model_name.startswith('linear_efficientnet-'):
        # freeze the base vectors, and train the final linear layer
        base_model_name = args.model_name[len('linear_'):]
        if args.dataset == 'celeba' and args.target == 'all':
            predictor = nn.Linear(config.FEATURE_DIM[base_model_name], len(dataset.ALL_CELEBA_ATTRS))
            if load_pretrained:
                path = osp.join(config.MODELS_DIR, '{}_{}.pth'.format(dataset_name, args.model_name))
                widgets.load_states(predictor, path)
        else:
            raise RuntimeError("unrecognized dataset (target): {}({})".format(args.dataset, args.target))
    elif args.model_name == 'resnet152':
        predictor = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V2)
    elif args.model_name == 'swin-b':
        predictor = torchvision.models.swin_b(weights=torchvision.models.Swin_B_Weights.IMAGENET1K_V1)
    elif args.model_name == 'mobilenet-v3':
        predictor = torchvision.models.mobilenet_v3_large(weights=torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    elif args.model_name == 'convnext':
        predictor = torchvision.models.convnext_large(weights=torchvision.models.ConvNeXt_Large_Weights.IMAGENET1K_V1)
    elif args.model_name == 'femnist-cnn':
        predictor = FemnistCNN()
        if load_pretrained:
            predictor.load_state_dict(torch.load(osp.join(config.MODELS_DIR, 'femnist-cnn.pth'),
                                                 map_location=config.DEVICE))
    elif args.model_name == 'celeba-cnn':
        assert args.target != 'all', "celeba-cnn does not support all attributes"
        predictor = CelebaCNN()
        if load_pretrained:
            if args.target != 'Smiling':
                model_name = 'celeba-cnn_{}.pth'.format(args.target)
            else:
                model_name = 'celeba-cnn.pth'
            predictor.load_state_dict(torch.load(osp.join(config.MODELS_DIR, model_name),
                                                 map_location=config.DEVICE))
    elif args.model_name == 'c3d':
        predictor = C3D(num_classes=config.NUM_CLASSES[args.dataset])
        if load_pretrained:
            predictor.load_state_dict(torch.load(osp.join(config.MODELS_DIR, '{}_c3d.pth'.format(args.dataset)),
                                                 map_location=config.DEVICE))
    else:
        raise ValueError('Unsupported model_name: {}'.format(args.model_name))

    predictor = predictor.to(config.DEVICE)
    return predictor


def parse_base_head(args: argparse.Namespace) -> nn.Linear:
    """parse the final linear layer of the base model (pretrained)"""
    predictor = parse_base_model(args, load_pretrained=True)

    # return the final classifier layer
    if args.model_name.startswith('efficientnet-'):
        classifier = predictor._fc
    elif args.model_name.startswith('linear_efficientnet-'):
        classifier = predictor
    elif args.model_name == 'resnet152':
        classifier = predictor.fc
    elif args.model_name == 'swin-b':
        classifier = predictor.head
    elif args.model_name == 'mobilenet-v3':
        classifier = predictor.classifier[3]
    elif args.model_name == 'convnext':
        classifier = predictor.classifier[2]
    elif args.model_name == 'femnist-cnn':
        classifier = predictor.fc2
    elif args.model_name == 'celeba-cnn':
        classifier = predictor.classifier
    elif args.model_name == 'c3d':
        classifier = predictor.fc8
    else:
        raise ValueError('Unsupported model_name: {}'.format(args.model_name))
    classifier = classifier.to(config.DEVICE)

    return classifier


def parse_criterion(args: argparse.Namespace):
    if args.dataset == 'celeba' and args.target == 'all':
        # multi-label classification
        criterion = nn.BCEWithLogitsLoss().to(config.DEVICE)
        return criterion
    else:
        return nn.CrossEntropyLoss().to(config.DEVICE)


def parse_metrics(args: argparse.Namespace):
    if args.dataset == 'celeba' and args.target == 'all':
        # multi-label classification
        return widgets.ALL_METRICS_MULTI_LABEL
    else:
        return widgets.ALL_METRICS
