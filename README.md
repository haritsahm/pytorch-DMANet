<div align="center">

# Deep Multi-Branch Aggregation Network for Real-Time Semantic Segmentation in Street Scenes

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.2203.04037-B31B1B.svg)](https://arxiv.org/abs/2203.04037)

</div>

![DMA-Net Architecture](docs/dmanet-arch.png)

This is an implementation of DMA-Net in Pytorch. The project is for my self exploration with Pytorch Lightning and Hydra tools and enhance my programming skills. DMA-Net is a real-time semantic segmentation network for street scenes in self-driving cars.

## Added Features

1. **D-Adaptaion Optmizers**
   Learning rate free learning for SGD, AdaGrad and Adam! by [facebookresearch/dadaptation/](https://github.com/facebookresearch/dadaptation)
   Simlply enable by using:

   ```
   model.auto_lr=True model.lr=1.0
   ```

2. **Hyperparameter Search**
   Since its hard to reproduce the result from the original author, I added 2 variables `high_level_features` and `low_level_features` to set the feature sizes in the model.

   - **high_level_features**: its the CBR (upmid_cbr) input size after addition ops between sub-network 3 and sub-network 4 in the upscaling pipeline.

   - **low_level_features**: its the CBR (uplow_cbr) input size after addition ops between sub-network 2 and upmid_cbr in the upscaling pipeline.

   ```
   model.net.low_level_features=128 model.net.high_level_features=128
   ```

# How to run

## Install dependencies

```bash
# clone project
git clone https://github.com/haritsahm/pytorch-DMANet.git
cd pytorch-DMANet

# [OPTIONAL] create conda environment
conda create -n myenv python=3.10
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

## Prepare dataset

Run and follow the [notebook](notebooks/dataset-preparation.ipynb) to prepare and visualize dataset using Fiftyone
![Fiftyone Sample](docs/fiftyone-sample.png)

## Train Commands

### 1. Train with default configurations

```bash
# train on CPU
python src/train.py trainer=cpu paths.data_dir=data/cityscape_fo_image_segmentation

# train on GPU
python src/train.py trainer=gpu paths.data_dir=data/cityscape_fo_image_segmentation

# train with DDP (Distributed Data Parallel) (4 GPUs)
python src/train.py trainer=ddp trainer.devices=4 paths.data_dir=data/cityscape_fo_image_segmentation
```

### 2. Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
# train using cityscape dataset
python train.py experiment=dmanet_cityscape paths.data_dir=data/cityscape_fo_image_segmentation

# train using camvid dataset
python train.py experiment=dmanet_camvid
```

### 3. Override any parameter

```bash
python train.py experiment=dmanet_cityscape paths.data_dir=data/cityscape_fo_image_segmentation trainer.max_epochs=20 datamodule.batch_size=64 model.net.low_level_features=128 model.net.high_level_features=256
```

Read the full [documentation](docs/DOCS.md) on how to use pytorch-lightning + hydra

## TODO:

- [ ] Train model using cloud instances
- [ ] Validate and compare model metrics (cityscapes and camvid)
