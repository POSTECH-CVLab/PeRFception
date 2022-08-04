# PeRFception - Perception using Radiance Fields


### Quick Access

[Project Page]() | [Paper](https://openreview.net/pdf?id=MzaPEKHv-0J) | [Supplementary Materials](https://openreview.net/attachment?id=MzaPEKHv-0J&name=supplementary_material) | [Quick Video]() | [Description Video](teaser) | [Dataset]()

### Author Info

- [Yoonwoo Jeong](https://yoonwooinfo.notion.site) [[Google Scholar](https://scholar.google.com/citations?user=HQ1PMggAAAAJ&hl=en)]
- [Seungjoo Shin](https://seungjooshin.github.io/) [[Google Scholar]()]
- [Junha Lee](https://junha-l.github.io/) [[Google Scholar](https://scholar.google.com/citations?user=RB7qMm4AAAAJ&hl=ko)]
- [Chris Choy](https://chrischoy.org) [[Google Scholar](https://scholar.google.com/citations?user=2u8G5ksAAAAJ&hl=en&oi=ao)]
- [Anima Anandkumar](http://tensorlab.cms.caltech.edu/users/anima/) [[Google Scholar](https://scholar.google.com/citations?user=bEcLezcAAAAJ&hl=en&oi=ao)]
- [Minsu Cho](http://cvlab.postech.ac.kr/~mcho/) [[Google Scholar](https://scholar.google.com/citations?user=5TyoF5QAAAAJ&hl=en&oi=ao)]
- [Jaesik Park](http://jaesik.info/) [[Google Scholar](https://scholar.google.com/citations?user=_3q6KBIAAAAJ&hl=en&oi=ao)]

![alt text](assets/figure1.png)

## Abstract

The recent progress in implicit 3D representation, i.e., Neural Radiance Fields (NeRFs), has made accurate and photorealistic 3D reconstruction possible in a differentiable manner. This new representation can effectively convey the information of hundreds of high-resolution images in one compact format and allows photorealistic synthesis of novel views. In this work, using the variant of NeRF called Plenoxels, we create the first large-scale implicit representation datasets  for perception tasks, called PeRFception, which consists of two parts that incorporate both object-centric and scene-centric scans for classification and segmentation. It shows a significant memory compression rate (96.4%) from the original dataset, while containing both 2D and 3D information in a unified form. We construct the  classification and segmentation models that directly take as input this implicit format and also propose a novel augmentation technique to avoid overfitting on backgrounds of images. The code and data will be publicly available. 

## Downloading PeRFception-Datasets
-------
### Co3D (1.3TB total) [[download full(1.3TB)]()] [[download toy(TBD)]()]

|Dataset| # Scenes | # Frames | 3D Shape | Features | 3D-BKGD | Memory | Memoery(Rel)
|-|-|-|-|-|-|-|-|
|Co3D| 18.6K | 1.5M | pcd | C | X | 1.44TB | $$\pm0.00\%$$
|PeRFception-Co3D| 18.6K | $$\infty$$ | voxel | SH + D | O | 1.33TB | $$-6.94\%$$

We provide a link to download PeRFception-Co3D dataset. In addition, we also provide a toy dataset of PeRFception-Co3D dataset, which has only one scene for each class. You can also view several examples of our dataset on the demo of project page [[link]()]. 

------
### ScanNet (35GB total) [[download full (35GB)]()] [[download toy (TBD)]()]

|Dataset| # Scenes | # Frames | 3D Shape | Features | 3D-BKGD | Memory | Memoery(Rel)
|-|-|-|-|-|-|-|-|
|ScanNet| 1.5K | 2.5M | pcd | C | X | 966GB | $$\pm0.00\%$$
|PeRFception-ScanNet| 1.5K | $$\infty$$ | voxel | SH + D | O | 35GB | $$-96.4\%$$

We provide a link to download PeRFception-ScanNet dataset.  In addition, we also provide a toy dataset of PeRFception-ScanNet dataset, which has 20 scenes total. You can also view several examples of our dataset on the project page [[link]()]. 


## Get Ready (Installation)

Our code is verified on Ubuntu 20.04 with a CUDA version 11.1.  

```
conda create -n perfception -c anaconda python=3.8
conda activate perfception
(perfception) $conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip3 install imageio tqdm requests configargparse scikit-image imageio-ffmpeg piqa wandb pytorch_lightning==1.6.0 opencv-python gin-config gdown
```


## Demo 
We provide a short demo for rendering a scene on Co3D or ScanNet. After installing the requirements, you could run the demo with the codes below:
```
# Co3D demo
python3 -m run --ginc configs/co3d.gin
# ScanNet demo
python3 -m run --ginc configs/scannet.gin
```

## Rendering Co3D and ScanNet 
We deliver the full code to reproduce the performance reported in the main paper. To run the code, you should first put the dataset on a proper location. 

```
data
  |
  |--- co3d
         -- apple 
         -- banana
         ... 
  |
  |--- scannet
         -- scene000_00
         -- scene000_01
         ...
```
ScanNet-v2 can be downloaded in [here](http://www.scan-net.org/) and Co3D-v1 can be downloaded in [here](https://github.com/facebookresearch/co3d). Thanks to great functions in `wandb`, we could manage tremendous scripts. You can download the `sweep` file [here](TBD). 


## Downstream Tasks

### 2D object classification (PeRFception-Co3D)

We benchmark several 2D classification models on rendered PeRFception-Co3D. For faster reproducing, we also provide the rendered images from PeRFception-Co3D on the link [link](). Before running the code, be sure that you had put the  downloaded dataset on `data/perfcepton_2d`. You can easily reproduce the scores using the scripts of `scripts/downstream/2d_cls/[model].sh`. Details for the training pipeline and models are elaborated in the main paper. 

The pretrained models can be reached with the links below: 
[2d_score](assets/2D_score.png)

For recent updates, you can refer to the leaderboard link [here]().

### 3D object classification (PeRFception-Co3D)

We also benchmark several 3D classification models on PeRFception-Co3D. You can easily reproduce the scores using the scripts of `scripts/downstream/2d_cls/[model].sh`. Details for the training pipeline and models are elaborated in the main paper. 


### 3D semantic segmentation (PeRFception-ScanNet)
In PeRFception-ScanNet, we have evaluated several 3D semantic segmentation models with depth-supervised labels. 

## Plans for v2

According to the official Co3D repository[[link](https://github.com/facebookresearch/co3d)], authors provided an improved version, v2, of Co3D, which would result in better rendering quality and more accurate geometries in our model. We are planning to extend this work to PeRFception-Co3D-v2 from the Co3D-v2. 

## Acknowledgement
We appreciate for the reviewers for their constructive comments and suggestions. 