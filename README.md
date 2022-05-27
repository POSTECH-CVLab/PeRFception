# NeRF-Factory

## Motivation
We are living on the generation of NeRF. Researchers in NeRF communities have difficulties of fairly comparing NeRF models.
Since the evaluation steps of NeRF models are similar and often share the dataset, this project attempts to collect various NeRF models for convenient evaluation. 
Currently, we only support the PyTorch framework but planning to enable Jax supports.

## Contributor
This project is created and maintained by Yoonwoo Jeong, Jinoh Cho, and Seungjoo Shin.

## Available Modules
- JaxNeRF (Torch Version) 
    - Several options are different from the original NeRF (TensorFlow)
    - NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis: [[Code](https://github.com/google-research/google-research/tree/master/jaxnerf) | [Paper](https://arxiv.org/pdf/2003.08934) | [Arxiv](https://arxiv.org/abs/2003.08934)]
    - Document: [Link](./docs/jaxnerf.md)
- SNeRG (Torch)
    - Coming Soon
- MipNeRF (Torch)
    - Coming Soon
- MipNeRF360 (Torch)
    - Coming Soon
- NeRF++ (Torch)
    - Coming Soon
- Plenoxel (Torch)
    - Reorganized the code structure for easier modification. 
    - Plenoxels: Radiance Fields without Neural Networks [[Code](https://github.com/sxyu/svox2) | [Paper](https://arxiv.org/abs/2112.05131) | [Arxiv](https://arxiv.org/pdf/2112.05131)]
    - Document: [Link](./docs/plenoxel.md)

## Dataset
```
```


## Acknowledgement

We have reffered to and borrowed the implementations of 
- PyTorch-NeRF (https://github.com/yenchenlin/nerf-pytorch)
- Jax-NeRF (https://github.com/google-research/google-research/tree/master/jaxnerf)
- Jax-SNeRG (https://github.com/google-research/google-research/tree/master/snerg)

## Requirements
```
conda create -n nerf_factory -c anaconda python=3.8
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip3 install imageio tqdm requests configargparse scikit-image imageio-ffmpeg piqa wandb pytorch_lightning==1.6.0 opencv-python gin-config gdown
```
