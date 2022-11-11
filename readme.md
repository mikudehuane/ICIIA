# Configuration
Please follow the instructions below to configure the project.
## Create Python Environment
Run the following commands to create the desired python environment
```bash
conda create -n iciia python=3.7.13
conda activate iciia
pip install -r requirements.txt
```
## Download the Datasets
After downloading, [data](data) folder should looks like:
```
data
├── ImageNet/
|   ├── ILSVRC
|   │   ├── Annotations
|   │   │   └── CLS-LOC
|   │   │       ├── train
|   │   │       └── val
|   │   ├── Data
|   │   │   └── CLS-LOC
|   │   │       ├── ILSVRC2012_devkit_t12
|   │   │       ├── ILSVRC2012_devkit_t12.tar.gz
|   │   │       ├── meta.bin
|   │   │       ├── move_val.py
|   │   │       ├── test
|   │   │       ├── train
|   │   │       └── val
|   │   └── ImageSets
|   │       └── CLS-LOC
|   │           ├── test.txt
|   │           ├── train_cls.txt
|   │           ├── train_loc.txt
|   │           └── val.txt
|   ├── LOC_sample_submission.csv
|   ├── LOC_synset_mapping.txt
|   ├── LOC_train_solution.csv
|   └── LOC_val_solution.csv
├── inaturalist
|   ├── FedScale
|   |   └── client_data_mapping
|   │       ├── inaturalist.csv
|   │       ├── train.csv
|   │       └── val.csv
|   ├── kaggle_sample_submission.csv
|   ├── test2019
|   ├── test2019.json
|   ├── train2019.json
|   ├── train_val2019
|   └── val2019.json
├── UCF-101
|   ... (class folders)
|   ├── testlist01.txt
|   ├── testlist02.txt
|   ├── testlist03.txt
|   ├── trainlist01.txt
|   ├── trainlist02.txt
|   └── trainlist03.txt
└── LEAF
    └── data
        ├── celeba
        │   ├── data
        │   └── preprocess
        ├── femnist
        │   ├── data
        │   └── preprocess
        └── utils
```

### iNaturalist 2019
Please download from [Kaggle](https://www.kaggle.com/competitions/inaturalist-2019-fgvc6) to [data/inaturalist](data/inaturalist).
Download the client split from [FedScale](https://github.com/SymbioticLab/FedScale/tree/master/benchmark/dataset/inaturalist)

### FEMNIST and CelebA
We modify the code from [LEAF](https://github.com/TalwalkarLab/leaf) to include the post-process scripts.
Please go to [data/LEAF/data/femnist/preprocess](data/LEAF/data/femnist/preprocess) and type the following command to download and process the dataset.
```bash
./preprocess.sh -s niid --sf 1.0 -k 0 -t sample --smplseed 1651747626 --spltseed 1651750258
```

For CelebA, please follow the instructions on [LEAF](https://github.com/TalwalkarLab/leaf/tree/master/data/celeba) to download the raw data, and then cd [data/LEAF/data/celeba/preprocess](data/LEAF/data/celeba/preprocess) to run
```bash
./preprocess.sh -s niid --sf 1.0 -k 5 -t user --smplseed 1665910210 --spltseed 1665910211
```

### ImageNet
Please download from [Kaggle](https://www.kaggle.com/c/imagenet-object-localization-challenge/overview/description) to [data/ImageNet](data/ImageNet).
Run [move_val.py](data/ImageNet/ILSVRC/Data/CLS-LOC/move_val.py) to move the validation images to the corresponding folders.

### UCF101
Please download from [UCF](https://www.crcv.ucf.edu/data/UCF101.php) to [data/UCF101](data/UCF101).

# Pre-train the Global Model
## Pre-training Models
Due to the large model size, we provide only the commands to pre-train the model here.
Note that the ImageNet-1K pre-trained weights will be automatically downloaded by the scripts.

```bash
# iNaturalist 2019
python main.py --phase=pretrain -ds=inaturalist -mn=efficientnet-b0 -bs=32 -vbs=64 -tebs=64 --num_workers=4 -lr=0.01 --lr_decay=0.1 --momentum=0.9 --decay_every=10

# FEMNIST
python main.py --phase=pretrain -ds=femnist -mn=femnist-cnn -bs=32 -vbs=3200 -tebs=3200

# CelebA
python main.py --phase=pretrain -ds=celeba -mn=efficientnet-b0 -bs=32 -vbs=64 -tebs=64 --num_workers=4 --momentum=0.9 --target=all

# UCF101
# Please first download the pretrained model from https://github.com/jfzhang95/pytorch-video-recognition to models/ucf101_caffe.pth
python main.py --phase=pretrain -ds=ucf101 -mn=c3d -bs=32 -vbs=64 -tebs=64 --num_workers=4 -lr=0.01
```

Then, move log/pretrain/\`date\`/models/best.pth to 
- iNaturalist 2019: models/inaturalist_efficientnet-b0.pth
- FEMNIST: models/femnist-cnn.pth
- CelebA: models/celeba_efficientnet-b0.pth
- UCF101: models/ucf101_c3d.pth

## Extracting Features
```bash
# iNaturalist 2019
python prep_feats.py -mn=efficientnet-b0 -ds=inaturalist -bs=32 --num_workers=4

# FEMNIST
python prep_feats.py -mn=femnist-cnn -ds=femnist -bs=3200

# CelebA
python prep_feats.py -ds=celeba --target=all -mn=efficientnet-b0 -bs=64 --num_workers=4

# ImageNet-1K
python prep_feats.py -ds=imagenet -mn=`NAME` -bs=32 --num_workers=4

# UCF101
python prep_feats.py -ds=ucf101 -mn=c3d -bs=4 --num_workers=4
```

# Experiments

## Number of Partitions
Run the following commands to train ICIIA with different number of layers and partitions.

```bash
# iNaturalist 2019
python main.py -nh=4 -ds=inaturalist --split=user -mn=efficientnet-b0 -nl=`N` -np=`P` -bs=16 --seed=`SEED`

# ImageNet-1K
python main.py -nh=4 -ds=imagenet --split=sample -mn=efficientnet-b4 -nl=`N` -np=`P` -bs=16 --seed=`SEED` --soft_reserve_ratio=1.0
```

## Level of Cross-Client Class Heterogeneity
Run the following commands to train ICIIA with different level of cross-client class heterogeneity.

```bash
python main.py -nh=4 -nl=3 -np=1 -mn=efficientnet-b4 --soft_reserve_ratio=`RATIO` --seed=`SEED`
```

## Number of Historical Samples
Please first copy the trained ICIIA modules in log/head_per/\`date\`/models/best.pth to
- N=2, P=1, SEED=0: inaturalist_efficientnet-b0_2-1p-s0.pth
- N=2, P=1, SEED=1: inaturalist_efficientnet-b0_2-1p-s1.pth
- N=2, P=1, SEED=2: inaturalist_efficientnet-b0_2-1p-s2.pth
- N=2, P=256, SEED=0: inaturalist_efficientnet-b0_2-256p-s0.pth
- N=2, P=256, SEED=1: inaturalist_efficientnet-b0_2-256p-s1.pth
- N=2, P=256, SEED=2: inaturalist_efficientnet-b0_2-256p-s2.pth
```bash
# ICIIA-B
python adapt.py --seed=`SEED`

# ICIIA-T
python adapt.py -np=256 --seed=`SEED`
```

## Comparison with Baselines
Run the following commands to train ICIIA or the baselines
```bash
# iNaturalist 2019
## ICIIA-B
python main.py -nh=4 -ds=inaturalist --split=user -mn=efficientnet-b0 -nl=2 -np=1 -bs=16 --seed=`SEED`
## ICIIA-T
python main.py -nh=4 -ds=inaturalist --split=user -mn=efficientnet-b0 -nl=2 -np=256 -bs=16 --seed=`SEED`

# FEMNIST
## ICIIA-B
python main.py -ds=femnist -nh=4 -nl=1 -mn=femnist-cnn -np=1 --seed=`SEED`
## ICIIA-T
python main.py -ds=femnist -nh=4 -nl=1 -mn=femnist-cnn -np=256 --seed=`SEED`
## Fine-Tuning
python finetune.py -ds=femnist -mn=femnist-cnn
## Prompt Tuning
python main.py -ds=femnist -mn=femnist-cnn -tl=prompt --dim_prompt=1024 --seed=`SEED`

# CelebA
## ICIIA-B
python main.py -nh=4 -ds=celeba -mn=efficientnet-b0 -nl=1 -np=1 -bs=0 --eval_mode=batch --split=user --target=all --seed=`SEED`
## ICIIA-T
python main.py -nh=4 -ds=celeba -mn=efficientnet-b0 -nl=1 -np=256 -bs=0 --eval_mode=batch --split=user --target=all --seed=`SEED`

# ImageNet-1K
## ICIIA-B
python main.py -nh=4 -nl=3 -np=1 -mn=`NAME` --soft_reserve_ratio=1.0 --seed=`SEED`
## ICIIA-T
python main.py -nh=4 -nl=3 -np=256 -mn=`NAME` --soft_reserve_ratio=1.0 --seed=`SEED`
## Fine-Tuning
python finetune.py -mn=`NAME`
## Prompt Tuning
python main.py -nh=4 -mn=`NAME` --soft_reserve_ratio=1.0 -tl=prompt --dim_prompt=`D/2` --seed=`SEED`

# UCF101
## ICIIA-B
python main.py -nh=4 -ds=ucf101 -mn=c3d -nl=2 -np=1 -bs=16 --seed=`SEED`
## ICIIA-T
python main.py -nh=4 -ds=ucf101 -mn=c3d -nl=2 -np=256 -bs=16 --seed=`SEED`
## Fine-Tuning
python finetune.py -ds=ucf101 -mn=c3d
## Prompt Tuning
python main.py -ds=ucf101 -mn=c3d -tl=prompt --dim_prompt=2048 --seed=`SEED`
```

## Ablation Study
- Supply -tl=ff to run without attention.
- Supply -np=\`D\` to run without projection.
- Supply --no_transpose to run without feature shuffling.

# Results
We have placed the raw excel tables of the results in [results](results).
