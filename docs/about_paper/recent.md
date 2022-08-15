---
layout: default
title: "Why Plenoxels?"
parent: About Paper
nav_order: 3
---

# Why Plenoxels?

There are several projects that reduce training time of optimizing neural fields, such as [DVGO-v1](https://arxiv.org/abs/2111.11215), [DVGO-v2](https://arxiv.org/abs/2206.05085), [Plenoxels](https://arxiv.org/abs/2112.05131), [Instant Neural Graphics Primitives](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf), [TensoRF](https://arxiv.org/abs/2203.09517), and [PointNeRF](https://arxiv.org/abs/2201.08845). Here, we compare recent fast-training NeRF models and describe why Plenoxels is a suitable representation for perception datasets.

## Reason 1: Plenoxels is a fully explicit representation

According to the second version of DVGO paper, [Improved Direct Voxel Grid Optimization](https://arxiv.org/abs/2206.05085), Plenoxels is the only representation that uses explicit features only. In other words, Plenoxels directly stores density volume, and view-dependent colors by spherical harmonics coefficients. 

[Reference: DVGO-v2](https://arxiv.org/pdf/2206.05085.pdf)

| Method | Data structure | Density | Color | Training Time |
|:-:|:-:|:-:|:-:|:-:|
| [DVGO](https://arxiv.org/abs/2111.11215) | Dense Grid | **Explicit** | Hybrid | < 30min 
| [DVGO-v2](https://arxiv.org/abs/2206.05085) | Dense Grid | **Explicit** | Hybrid | < 20min 
| [Plenoxels](https://arxiv.org/abs/2112.05131) | Sparse Grid | **Explicit** | **Explicit** | < 30min
| [INGP](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf) | Multi-level Hash | Hybrid | Hybrid | < 5min
| [TensoRF](https://arxiv.org/abs/2203.09517) | Dense Grid | **Explicit** | Hybrid | < 30min
| [PointNeRF](https://arxiv.org/abs/2201.08845) | Point Cloud | **Explicit** | **Explicit** | > 1 day

## Reason 2: Great reconstruction quality

Plenoxels shows great ability for reconstructing scenes compared to the others in both indoor and outdoor scenarios. We randomly pick 5 sequences each from CO3D and ScanNet. We report the rendering quality and training time for each method. We compare Plenoxels with DVGO-v2 since it has shown comparable performance on outdoor scenes. For the other methods, we could not use them as our data format since 1) INGP implicitly encodes geometries and does not cover unbound scenarios, 2) TensoRF and DVGO-v1 do not have representation for backgrounds, and 3) PointNeRF takes a long time for optimization. For DVGO-v2, we follow the Tanks and Temples setup.

| Method | PSNR | SSIM | LPIPS |
|:-:|:-:|:-:|:-:|
| [DVGO-v2](https://arxiv.org/abs/2206.05085) | 18.11 | 0.6368 | 0.5957 |
| [Plenoxels](https://arxiv.org/abs/2112.05131) | 30.81 | 0.8551 | 0.3353 |

Compared to DVGO-v2, Plenoxels was less senstive to the domain of scenes. We also attach qualitative results in the figure below. 

<img src="../../../assets/images/dvgo.jpg">
