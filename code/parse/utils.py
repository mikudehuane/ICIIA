import random
from typing import Iterable, Tuple, Union

import numpy as np
import torch
import torchvision
from efficientnet_pytorch import EfficientNet
from torch.utils.data import Dataset, TensorDataset, Subset
import os.path as osp

from torchvision import transforms
from torchvision.transforms import InterpolationMode

import config
import dataset
import logging

import widgets

_logger = logging.getLogger(__name__)


def get_dataset_name(dataset_name: str, target: str = '') -> str:
    """get the real dataset name, which will be used in model path"""
    if dataset_name == 'celeba' and target != 'Smiling':
        # for celeba, the name is coupled with target
        # the default celeba means target='Smiling' (LEAF setting)
        dataset_name = 'celeba-{}'.format(target)

    return dataset_name


def get_head_dataset(dataset_name, model_name, *, target='', need_features=True) -> Union[
        TensorDataset, Tuple[TensorDataset, TensorDataset]]:
    """get the dataset for head model (loaded features from config.CACHE_DIR)

    This function is responsible for data loading, not for data splitting

    Args:
        dataset_name: args.dataset
        model_name: args.model_name
        target: args.target
        need_features: need features to be loaded

    Returns:
        imagenet, celeba: (dataset_train_val, dataset_test)
        femnist: dataset_full
    """
    # For celeba dataset and efficientnet-b7 model, we use pretrained features directly, which are stored in
    # celeba-all dir. Thus, we need to change the dataset_name to celeba-all to load it.
    is_hacked_with_all = dataset_name == 'celeba' and model_name == 'efficientnet-b7' and target != 'all'

    # parse paths
    if model_name.startswith('linear_'):
        model_name = model_name[len('linear_'):]

    if dataset_name == 'imagenet':
        # load user-data index
        dset_folder = osp.join(config.DATA_IMAGENET_CACHE_DIR, model_name)
    else:
        if is_hacked_with_all:
            dataset_name = 'celeba-all'
        else:
            dataset_name = get_dataset_name(dataset_name, target)
        dset_folder = osp.join(config.CACHE_FEATURE_DIR, dataset_name, model_name)
    # now, dataset_name is the parent folder name in config.CACHE_FEATURE_DIR
    # - for celeba, may be tailed with target
    # - for celeba and efficientnet-b7, may be forced with '-all' tail
    # model_name is the sub-folder name.

    # load data
    if dataset_name == 'femnist':
        all_features, all_labels = dataset.load_features(dset_folder, need_features=need_features)
        full_dataset = TensorDataset(all_features, all_labels)
        return full_dataset
    else:
        # raw dataset
        train_val_features, train_val_labels = dataset.load_features(
            osp.join(dset_folder, 'train'), need_features=need_features)
        test_features, test_labels = dataset.load_features(osp.join(dset_folder, 'val'), need_features=need_features)
        if is_hacked_with_all:  # all features should be loaded
            assert train_val_labels.size(1) == len(dataset.CELEBA_ATTR_NAMES)
            assert test_labels.size(1) == len(dataset.CELEBA_ATTR_NAMES)

        # process data optionally
        if dataset_name == 'celeba-all':
            if is_hacked_with_all:
                # all features loaded, but use only the required label
                target_idx = dataset.CELEBA_ATTR_NAMES.index(target)
                train_val_labels = train_val_labels[:, target_idx]
                test_labels = test_labels[:, target_idx]
            elif model_name == 'efficientnet-b7':  # only for efficientnet-b7, the dumped labels are the full set
                # fetch the used labels
                train_val_labels = dataset.get_celeba_all_labels(train_val_labels)
                test_labels = dataset.get_celeba_all_labels(test_labels)
            else:
                train_val_labels = train_val_labels.to(dtype=torch.float32)
                test_labels = test_labels.to(dtype=torch.float32)

        # convert to tensor dataset
        dataset_train_val = TensorDataset(train_val_features, train_val_labels)
        dataset_test = TensorDataset(test_features, test_labels)

        return dataset_train_val, dataset_test


def get_base_dataset(
        dataset_name: str, *,
        model_name: str, train_ratio: float = None, target: str = ''
) -> Union[Tuple[Dataset, Dataset], Tuple[Dataset, Dataset, Dataset]]:
    """get the dataset with raw features, train-val-test split or train-test split if train_ratio not specified"""

    if dataset_name in ('imagenet', 'inaturalist'):
        # transform for imagenet and inaturalist
        if model_name.startswith('efficientnet-'):
            # data preprocessing copied from
            # https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/examples/imagenet/main.py
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            image_size = EfficientNet.get_image_size(model_name)

            transform = transforms.Compose([
                transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ])
            del image_size, normalize
        elif model_name == 'resnet152':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            image_size = 224
            transform = transforms.Compose([
                transforms.Resize(image_size, interpolation=InterpolationMode.BILINEAR),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ])
            del image_size, normalize
        elif model_name == 'swin-b':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            transform = transforms.Compose([
                transforms.Resize(238, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
            del normalize
        elif model_name in ('mobilenet-v3', 'convnext'):
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            transform = transforms.Compose([
                transforms.Resize(232, interpolation=InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
            del normalize
        else:
            raise ValueError('Unsupported model_name: {}'.format(model_name))

        if dataset_name == 'imagenet':
            assert train_ratio is None, 'imagenet dataset does not support train_ratio (only prep_feats.py use this dset)'

            dataset_train_val = torchvision.datasets.ImageNet(
                config.DATA_IMAGENET_DIR,
                split='train',
                transform=transform
            )
            dataset_test = torchvision.datasets.ImageNet(
                config.DATA_IMAGENET_DIR,
                split='val',
                transform=transform
            )
            return dataset_train_val, dataset_test
        elif dataset_name == 'inaturalist':
            dataset_train_val = dataset.inaturalist.INaturalistDataset(
                file_index=config.DATA_INATURALIST_TRAIN_INDEX_PATH, transform=transform)
            dataset_test = dataset.inaturalist.INaturalistDataset(
                file_index=config.DATA_INATURALIST_TEST_INDEX_PATH, transform=transform)
            if train_ratio is not None:
                train_indices, val_indices = dataset.train_val_split(
                    list(range(len(dataset_train_val))), train_ratio=train_ratio, shuffle=True,
                    randomer=random.Random(0))
                dataset_train = Subset(dataset_train_val, train_indices)
                dataset_val = Subset(dataset_train_val, val_indices)
                return dataset_train, dataset_val, dataset_test
            else:
                return dataset_train_val, dataset_test
        else:
            raise AssertionError('unexpected dataset_name: {}'.format(dataset_name))
    elif dataset_name == 'femnist':
        _logger.info('loading all_data.pkl...')
        full_data = np.load(osp.join(config.DATA_FEMNIST_DIR, 'all_data.pkl'), allow_pickle=True)
        _logger.info('loaded')
        # load index
        full_index = widgets.load_json(osp.join(config.DATA_FEMNIST_DIR, 'all.json'))
        train_index = full_index['train']
        test_index = full_index['test']
        # create dataset
        dataset_test = dataset.FemnistDataset(test_index, data_dir=full_data)
        if train_ratio is not None:
            train_index, val_index = dataset.train_val_split(train_index, train_ratio=train_ratio, shuffle=True)
            dataset_train = dataset.FemnistDataset(train_index, data_dir=full_data)
            dataset_val = dataset.FemnistDataset(val_index, data_dir=full_data)
            return dataset_train, dataset_val, dataset_test
        else:
            dataset_train = dataset.FemnistDataset(train_index, data_dir=full_data)
            return dataset_train, dataset_test
    elif dataset_name == 'celeba':
        if model_name.startswith('efficientnet-'):
            image_size = EfficientNet.get_image_size(model_name)
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            transform = transforms.Compose([
                transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ])
        elif model_name == 'celeba-cnn':
            image_size = 84
            transform = transforms.Compose([
                transforms.Pad(padding=(20, 0), fill=0),
                transforms.Resize(image_size, interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ])
        else:
            raise ValueError('Unsupported model_name: {}'.format(model_name))

        dataset_train_val = dataset.CelebaDataset(
            user_index_path=config.DATA_CELEBA_TRAIN_INDEX_PATH, transform=transform, target_attr=target)
        dataset_test = dataset.CelebaDataset(user_index_path=config.DATA_CELEBA_TEST_INDEX_PATH, transform=transform,
                                             target_attr=target)
        if train_ratio is not None:
            train_indices, val_indices = dataset.train_val_split(list(range(len(dataset_train_val))),
                                                                 train_ratio=train_ratio, shuffle=True)
            dataset_train = Subset(dataset_train_val, train_indices)
            dataset_val = Subset(dataset_train_val, val_indices)
            return dataset_train, dataset_val, dataset_test
        else:
            return dataset_train_val, dataset_test
    elif dataset_name == 'ucf101':
        dataset_train_val = dataset.ucf101.UCF101Dataset(
            dataset.ucf101.VideoLoader(train=True), video_index=config.DATA_UCF101_TRAIN_INDEX_PATH)
        dataset_test = dataset.ucf101.UCF101Dataset(
            dataset.ucf101.VideoLoader(train=False), video_index=config.DATA_UCF101_TEST_INDEX_PATH)
        if train_ratio is not None:
            train_indices, val_indices = dataset.ucf101.train_val_split(
                dataset_train_val.video_index, train_ratio=train_ratio)
            dataset_train = dataset.ucf101.UCF101Dataset(
                dataset.ucf101.VideoLoader(train=True), video_index=train_indices)
            dataset_val = dataset.ucf101.UCF101Dataset(
                dataset.ucf101.VideoLoader(train=True), video_index=val_indices)
            return dataset_train, dataset_val, dataset_test
        else:
            return dataset_train_val, dataset_test
    else:
        raise ValueError('Unsupported dataset_name: {}'.format(dataset_name))
