import argparse
import os
import sys
from collections import Callable

import numpy as np
import torch
import torchvision
from PIL import Image
from efficientnet_pytorch import EfficientNet
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os.path as osp

import dataset
import parse
import widgets
from model import FemnistCNN

from torchvision.transforms import InterpolationMode
from tqdm import tqdm

import config
from parse import parse_base_model


def extract_features(dataset: Dataset, func: Callable, output_dir: str, *,
                     batch_size=8, num_workers=3):
    os.makedirs(output_dir, exist_ok=True)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    all_features = None
    start_idx: int = 0
    with open(osp.join(output_dir, 'labels.txt'), 'w') as label_f:
        for inputs, labels in tqdm(loader):
            inputs = inputs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            for label in labels:
                if label.dim() == 0:
                    label_f.write(str(label.item()) + '\n')
                elif label.dim() == 1:
                    label_f.write(','.join(map(lambda x: str(int(x + 1e-7)), label.tolist())) + '\n')
                else:
                    raise RuntimeError('label dim not supported: {}'.format(label.dim()))
            features = func(inputs)
            features = features.cpu().detach().numpy()
            if all_features is None:
                all_features = np.zeros((len(dataset), features.shape[1]), dtype=np.float32)
            all_features[start_idx: start_idx + features.shape[0]] = features
            start_idx += int(features.shape[0])
    np.save(osp.join(output_dir, 'features.npy'), all_features, allow_pickle=False)


def config_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mn', '--model_name', type=str, default='efficientnet-b2')
    parser.add_argument('-bs', '--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('-ds', '--dataset', type=str, default='imagenet',
                        choices=('imagenet', 'femnist', 'celeba', 'ucf101', 'inaturalist'),
                        help='dataset to extract features')
    parser.add_argument('--target', default='Smiling', type=str,
                        help='target attribute to predict (for celeba)')
    parser.add_argument("--num_workers", default=0, type=int,
                        help="number of workers to load data")
    args = parser.parse_args()

    assert 'train_ratio' not in args.__dict__

    return args


def main():
    args = config_args()

    if args.device is not None:
        config.DEVICE = torch.device(args.device)

    predictor = parse_base_model(args)

    # conditioned by args.model_name: define get_features function
    if args.model_name.startswith('efficientnet-'):
        def get_features(x):
            x = predictor.extract_features(x)
            x = predictor._avg_pooling(x)
            x = x.flatten(start_dim=1)
            return x
    elif args.model_name == 'resnet152':
        def get_features(x):
            # copied from resnet class, but without the final fc
            x = predictor.conv1(x)
            x = predictor.bn1(x)
            x = predictor.relu(x)
            x = predictor.maxpool(x)

            x = predictor.layer1(x)
            x = predictor.layer2(x)
            x = predictor.layer3(x)
            x = predictor.layer4(x)

            x = predictor.avgpool(x)
            x = torch.flatten(x, 1)
            return x
    elif args.model_name.startswith('swin-'):
        def get_features(x):
            x = predictor.features(x)
            x = predictor.norm(x)
            x = x.permute(0, 3, 1, 2)
            x = predictor.avgpool(x)
            x = torch.flatten(x, 1)
            return x
    elif args.model_name == 'mobilenet-v3':
        def get_features(x):
            x = predictor.features(x)
            x = predictor.avgpool(x)
            x = torch.flatten(x, 1)
            x = predictor.classifier[0](x)
            x = predictor.classifier[1](x)
            x = predictor.classifier[2](x)
            return x
    elif args.model_name == 'convnext':
        def get_features(x):
            x = predictor.features(x)
            x = predictor.avgpool(x)
            x = predictor.classifier[0](x)
            x = predictor.classifier[1](x)
            return x
    elif args.model_name in ('femnist-cnn', 'celeba-cnn', 'c3d'):
        def get_features(x):
            return predictor.extract_features(x)
    else:
        raise ValueError('Unsupported model_name: {}'.format(args.model_name))

    predictor.to(device=config.DEVICE)
    predictor.eval()

    if args.dataset == 'imagenet':
        train_dataset, val_dataset = parse.get_base_dataset(args.dataset, model_name=args.model_name)
        cache_dir = osp.join(config.DATA_IMAGENET_CACHE_DIR, args.model_name)
        extract_features(val_dataset, get_features,
                         osp.join(cache_dir, 'val'), batch_size=args.batch_size)
        extract_features(train_dataset, get_features,
                         osp.join(cache_dir, 'train'), batch_size=args.batch_size)

    elif args.dataset == 'femnist':
        cache_dir = osp.join(config.CACHE_FEATURE_DIR, args.dataset, args.model_name)
        full_dataset = dataset.FemnistDataset.get_full_dataset(osp.join(config.DATA_FEMNIST_DIR, 'all.json'),
                                                               osp.join(config.DATA_FEMNIST_DIR, 'all_data.pkl'))
        extract_features(full_dataset, get_features,
                         cache_dir, batch_size=args.batch_size)
    elif args.dataset == 'celeba':
        train_dataset, val_dataset = parse.get_base_dataset(
            args.dataset, model_name=args.model_name, target=args.target)

        dataset_name = parse.get_dataset_name(args.dataset, target=args.target)
        cache_dir = osp.join(config.CACHE_FEATURE_DIR, dataset_name, args.model_name)
        extract_features(val_dataset, get_features,
                         osp.join(cache_dir, 'val'), batch_size=args.batch_size)
        extract_features(train_dataset, get_features,
                         osp.join(cache_dir, 'train'), batch_size=args.batch_size)
    elif args.dataset == 'ucf101':
        train_dataset, val_dataset = parse.get_base_dataset(args.dataset, model_name=args.model_name)
        cache_dir = osp.join(config.CACHE_FEATURE_DIR, args.dataset, args.model_name)
        extract_features(val_dataset, get_features,
                         osp.join(cache_dir, 'val'), batch_size=args.batch_size)
        extract_features(train_dataset, get_features,
                         osp.join(cache_dir, 'train'), batch_size=args.batch_size)
    elif args.dataset == 'inaturalist':
        train_dataset, val_dataset = parse.get_base_dataset(args.dataset, model_name=args.model_name)
        cache_dir = osp.join(config.CACHE_FEATURE_DIR, args.dataset, args.model_name)
        extract_features(val_dataset, get_features,
                         osp.join(cache_dir, 'val'), batch_size=args.batch_size)
        extract_features(train_dataset, get_features,
                         osp.join(cache_dir, 'train'), batch_size=args.batch_size)
    else:
        raise ValueError('Unsupported dataset: {}'.format(args.dataset))


if __name__ == '__main__':
    main()
