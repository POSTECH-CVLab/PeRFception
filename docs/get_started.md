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
There are two multi-view datasets that our code supports: CO3D and ScanNet. We provide a script to automatically download CO3D-v1. We are currently planning to extend our PeRFception-Co3D dataset to the second version, which will be generated from Co3D-v2. Users can download CO3D-v1 by the script below:
```bash
python3 -m scripts.download_co3d_v1.py
```

To select specific classes, please add the argument as the following:
```bash
python3 -m scripts.download_co3d_v1.py --class_name "apple, banana"
```

Due to the limited access on ScanNet, we could not provide an automatic script for ScanNet. We recommend to follow the official instruction of [ScanNet](http://www.scan-net.org/) to get multi-view images of ScanNet. 

## Data Generation - Rendering a Specific Scene

Here, we provide a guide to reproduce our generated dataset. Before running the code, be sure that you have properly installed your requirements as suggested in the `Installation` section. For convenience, we have integrated the script format for CO3D and ScanNet as the following:

```bash
python3 -m run --ginc configs/[dataset].gin --scene_name [scene_name]

## Examples
python3 -m run --ginc configs/co3d.gin --scene_name 14_158_900
python3 -m run --ginc configs/scannet.gin --scene_name scene0000_00

```

## Data Generation - Rendering whole Scenes

We also provide a guide to manage render tremendously many frames to generate a whole dataset. Thanks to a great experiment management toolkit in [wandb](https://wandb.ai/site), called sweep, we could run whole scenes without significant efforts on distributing experiments to multiple devices. For convenience, we also provide sweep configuration files for both Co3D and ScanNet. We recommend users to read the [official documentation](https://docs.wandb.ai/guides/sweeps) of sweep function in wandb. You could download the configuration files in the link [here](). 

## Download the Generated Dataset

You could also download the generated dataset. As suggested by one of reviewers, we have moved our data cloud to OneDrive since it is reachable from any countries. In addition, we have split our dataset into chunks(11~15GB each) to handle network disconnection issues. If you desire to download whole datasets with a single command line script, you could use the command below.

```bash
python3 -m scripts.download_perf_co3d.py --outdir [directory]
python3 -m scripts.download_perf_scannet.py --outdir [directory]
```

If you desire to download a specific chunk, then you can use the script below:

```bash
python3 -m scripts.download_perf_co3d.py --outdir [directory] --chunk_number 00
python3 -m scripts.download_perf_scannet.py --outdir [directory] --chunk_number 00
```

Of course, you can download a specific scene with the script below:
```bash
python3 -m scripts.download_perf_co3d.py --outdir [directory] --scene_name 14_158_900
python3 -m scripts.download_perf_scannet.py --outdir [directory] --scene_name scene0000_00
```

## Benchmarking on Dataset

We share the code for reproducing performance on perception tasks on the main paper in the link [here](TBD). We are also planning to make a benchmark on several tasks. We'll soon be back with announcements.