---
layout: default
title: Get Started
permalink: /get_started
nav_order: 2
---

# Get Started
{: .no_toc }

Quick installation manual
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Installation

Users should first install several dependencies to run the code. Since our data generation codes strongly relies on CUDA operations of Plenoxels, it is recommended to use CUDA version higher than or equal to 11.0. If you were only interested on benchmarking, then the lower CUDA version would not cause problems. Our code is verified on Ubuntu 20.04 with a CUDA version 11.1. 

```bash
## Be sure CUDA >= 11.0
git clone https://github.com/POSTECH-CVLab/PeRFception.git
cd PeRFception
conda create -n perfception -c anaconda python=3.8 -y
conda activate perfception
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 \
    -c pytorch -c conda-forge -y
pip3 install imageio tqdm requests configargparse scikit-image imageio-ffmpeg piqa \
    wandb pytorch_lightning==1.5.5 opencv-python gin-config gdown
pip3 install .
```

## Preparing Datasets
There are two multi-view datasets that our code supports: CO3D and ScanNet. We are currently planning to extend our PeRFception-Co3D dataset to the second version, which will be generated from Co3D-v2. Users can download CO3D-v1 in the link [here](https://github.com/facebookresearch/co3d/tree/v1). To download ScanNet, we recommend to follow the official instruction of [ScanNet](http://www.scan-net.org/) to get multi-view images of ScanNet. 

## Data Generation - Rendering a Specific Scene

Here, we provide a guide to reproduce our generated dataset. Before running the code, be sure that you have properly installed your requirements as suggested in the `Installation` section. For convenience, we have integrated the script format for CO3D and ScanNet as the following:

```bash
python3 -m run --ginc configs/[dataset].gin --scene_name [scene_name]

## Examples
python3 -m run --ginc configs/co3d.gin --scene_name 14_158_900
python3 -m run --ginc configs/scannet.gin --scene_name scene0000_00
```

## Data Generation - Rendering whole Scenes

We also provide a guide to manage render tremendously many frames to generate a whole dataset. Thanks to a great experiment management toolkit in [wandb](https://wandb.ai/site), called sweep, we could run whole scenes without significant efforts on distributing experiments to multiple devices. For convenience, we also provide sweep configuration files for both Co3D and ScanNet. We recommend users to read the [official documentation](https://docs.wandb.ai/guides/sweeps) of sweep function in wandb. You could download the configuration files in the link [here](https://1drv.ms/u/s!As9A9EbDsoWcj6toSOfdeWMaHhqF3Q?e=1INfNg). 

## Download the Generated Dataset

You could also download the generated dataset. As suggested by one of reviewers, we have moved our data cloud to OneDrive since it is reachable from any countries. In addition, we have split our dataset into chunks(11~15GB each) to handle network disconnection issues. If you desire to download whole datasets with a single command line script, you could use the command below.

```
### Full download
python3 utils/download_perf.py --dataset co3d --outdir [outdir]
python3 utils/download_perf.py --dataset scannet --outdir [outdir]

### Specific Chunk
python3 utils/download_perf.py --dataset co3d --outdir [outdir] --chunks 77
python3 utils/download_perf.py --dataset co3d --outdir [outdir] --chunks [11, 22, 33]
```

## Benchmarking on Dataset

We share the code for reproducing performance on perception tasks on the main paper in the link [here](https://github.com/POSTECH-CVLab/NeRF-Downstream). We'll soon be back with announcements.