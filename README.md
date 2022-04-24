<div align="center">

# Deep Multi-Branch Aggregation Network for Real-Time Semantic Segmentation in Street Scenes

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.2203.04037-B31B1B.svg)](https://arxiv.org/abs/2203.04037)

</div>

## Description

![DMA-Net Architecture](docs/dmanet-arch.png)

This is an implementation of DMA-Net in Pytorch. The project is for my self exploration with Pytorch Lightning and Hydra tools and enhance my programming skills. DMA-Net is a real-time semantic segmentation network for street scenes in self-driving cars.

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/haritsahm/pytorch-DMANet.git
cd pytorch-DMANet

# [OPTIONAL] create conda environment
conda create -n myenv python=3.8
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

Train model with default configuration

```bash
# train on CPU
python train.py trainer.gpus=0

# train on GPU
python train.py trainer.gpus=1
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python train.py experiment=dma_net.yaml
```

You can override any parameter from command line like this

```bash
python train.py trainer.max_epochs=20 datamodule.batch_size=64
```

Track experiments with experiment trackers
```bash
python train.py trainer.max_epochs=20 datamodule.batch_size=64 logger=neptune
```

Read the full [documentation](docs/DOCS.md) on how to use pytorch-lightning + hydra
