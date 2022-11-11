import os.path as osp
import torch
import numpy as np
import random
import logging
import sys
import os

PROJECT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
CODE_DIR = osp.join(PROJECT_DIR, 'code')
CONFIG_DIR = osp.join(PROJECT_DIR, 'conf')
OUTPUTS_DIR = osp.join(PROJECT_DIR, 'outputs')
DATA_DIR = osp.join(PROJECT_DIR, 'data')

DATA_FEMNIST_DIR = osp.join(DATA_DIR, 'LEAF', 'data', 'femnist', 'data', 'clients_split')
DATA_CELEBA_ROOT_DIR = osp.join(DATA_DIR, 'celeba', 'data')
DATA_CELEBA_TRAIN_INDEX_PATH = osp.join(DATA_CELEBA_ROOT_DIR, 'train', 'all_data_0_0_keep_5_train_9.json')
DATA_CELEBA_TEST_INDEX_PATH = osp.join(DATA_CELEBA_ROOT_DIR, 'test', 'all_data_0_0_keep_5_test_9.json')
DATA_CELEBA_ATTR_PATH = osp.join(DATA_CELEBA_ROOT_DIR, 'raw', 'list_attr_celeba.txt')
DATA_UCF101_DIR = osp.join(DATA_DIR, 'UCF-101')
DATA_UCF101_TRAIN_INDEX_PATH = osp.join(DATA_UCF101_DIR, 'trainlist01.txt')
DATA_UCF101_TEST_INDEX_PATH = osp.join(DATA_UCF101_DIR, 'testlist01.txt')
DATA_UCF101_CLASS_INDEX_PATH = osp.join(DATA_UCF101_DIR, 'classInd.txt')
DATA_IMAGENET_DIR = osp.join(DATA_DIR, 'ImageNet', 'ILSVRC', 'Data', 'CLS-LOC')
DATA_IMAGENET_CACHE_DIR = osp.join(DATA_DIR, 'ImageNet', 'ILSVRC', 'Cache', 'CLS-LOC')
DATA_INATURALIST_DIR = osp.join(DATA_DIR, 'inaturalist')
DATA_INATURALIST_TRAIN_INDEX_PATH = osp.join(DATA_INATURALIST_DIR, 'FedScale', 'client_data_mapping', 'train.csv')
DATA_INATURALIST_TEST_INDEX_PATH = osp.join(DATA_INATURALIST_DIR, 'FedScale', 'client_data_mapping', 'val.csv')

LOG_DIR = osp.join(PROJECT_DIR, 'log')
MODELS_DIR = osp.join(PROJECT_DIR, 'models')
MODELS_UCF_PRETRAINED_PATH = osp.join(MODELS_DIR, 'ucf101-caffe.pth')

CACHE_DIR = osp.join(PROJECT_DIR, 'cache')
CACHE_FEATURE_DIR = osp.join(CACHE_DIR, 'features')

os.makedirs(LOG_DIR, exist_ok=True)

DEVICE = torch.device('cuda:0')

ALG_NAME = 'ICIIA'

SEED = 0


def config_seed(seed):
    global SEED
    torch.manual_seed(seed)
    # noinspection PyUnresolvedReferences
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    SEED = seed


config_seed(SEED)

logging_stream = None

FEATURE_DIM = {
    'efficientnet-b0': 1280,
    'efficientnet-b1': 1280,
    'efficientnet-b2': 1408,
    'efficientnet-b3': 1536,
    'efficientnet-b4': 1792,
    'efficientnet-b5': 2048,
    'efficientnet-b6': 2304,
    'efficientnet-b7': 2560,
    'linear_efficientnet-b7': 2560,
    'resnet152': 2048,
    'femnist-cnn': 2048,
    'celeba-cnn': 1152,
    'swin-b': 1024,
    'convnext': 1536,
    'mobilenet-v3': 1280,
    'c3d': 4096,
}
NUM_CLASSES = {
    'imagenet': 1000,
    'femnist': 62,
    'celeba': 2,
    'ucf101': 101,
    'inaturalist': 1010
}


def config_log(stream=sys.stdout, *, level=logging.DEBUG):
    # logging config
    logging.basicConfig(stream=stream, datefmt='%Y/%m/%d %H:%M:%S',
                        format='%(asctime)s [%(levelname)s] (%(name)s) - %(message)s',
                        level=level)
    global logging_stream
    logging_stream = stream
