# NeRF-Factory

## Motivation
We are living on the generation of NeRF. Researchers in NeRF communities have difficulties of fairly comparing NeRF models. Since the evaluation steps of NeRF models are similar and often share the dataset, this project attempts to collect various NeRF models for convenient evaluation. Currently, we only supports the PyTorch framework but planning to extend Jax supports.

## Available Modules
- JaxNeRF (Torch Version)
    - Several options are different from the original NeRF (TensorFlow)
- SNeRG (Torch Version)
    - PyTorch version of 
- MipNeRF (Torch Version)
    - Comming Soon
- MipNeRF360 (Torch Version)
    - Comming Soon 
- NeRF++ (Torch Version)
    - Comming Soon


## Acknowledgement

We have referred and borrowed the implementations of 
- PyTorch-NeRF ()
- Jax-NeRF ()
- Jax-SNeRG ()

## Requirements
```
conda create -n nerf_factory -c anaconda python=3.8
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip3 install imageio tqdm requests configargparse scikit-image imageio-ffmpeg piqa wandb pytorch_lightning==1.5.5 opencv-python
```