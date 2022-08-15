---
layout: default
title: "Memory-quality trade-off"
parent: About Paper
nav_order: 4
---

# Memory-quality Trade-off

We additionally explore quantization methods and resolution-memory trade-off to search for optimal configuration. At first, we have randomly sampled 5 scenes from Co3D. All the experiments will be added to our paper. We provide pre-trained checkpoints in the first column of each table.

## Experiment 1) Resolution-quality trade-off

We compare reconstruction qualities by varying the resolution: 64, 128, 256, and 512. We follow the setup from our paper, and estimate the required memory and reconstruction metrics (PSNR, SSIM, LPIPS).
 
| Resolution | Memory(MB/scene) | PSNR | SSIM | LPIPS |
|:-:|:-:|:-:|:-:|:-:|
| [64](https://1drv.ms/u/s!AgY2evoYo6FggrtOAzaUJlI15Oan4Q?e=lHgHjZ) | 38.0 | 26.56 | 0.7464 | 0.4827 |
| [128](https://1drv.ms/u/s!AgY2evoYo6Fggrwd3s_aEQMiqDy2vg?e=awmsXI) | 47.2 | 29.39 | 0.8081 | 0.4017 |
| [256](https://1drv.ms/u/s!AgY2evoYo6Fggr0Z9cV-P4IiQ67lsQ?e=pJtFef) | 63.2 | 30.81 | 0.8551 | 0.3353 |
| [384](https://1drv.ms/u/s!AgY2evoYo6Fggr1gYYtDQMtE66F1aQ?e=3kE36l) | 161.2 | 31.03 | 0.8619 | 0.3202 |

The setup with resolution 256 shows the best trade-off between memory and quality.
 
 
## Experiment 2) Quantization Methods

We further explore several quantization methods and appropriate compression bits that can be used for our method. **ours(SH)** applies our quantization on spherical harmonics coefficients, while **ours(SH+D)** applies our quantization on both spherical harmonics coefficients and densities. **Clipping** clips the feature values to heuristically searched interval; for instance, density should be non-negative values. We will elaborate thorough details about the compression methods in the supplementary materials. We additionally seek for the optimal bit for each compression method.

| Quantization Method | Bit| Memory | PSNR | SSIM | LPIPS |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| [ours(SH)](https://1drv.ms/u/s!AgY2evoYo6Fggr0Z9cV-P4IiQ67lsQ?e=pJtFef) | 8 | 63.2 | 30.81 | 0.8551 | 0.3353 |
| [ours(SH)](https://1drv.ms/u/s!AgY2evoYo6FggroUHVMSjyWBiZDQHw?e=XB7kow) | 4 | 37.4 | 27.48 | 0.7392 | 0.4512 |
| [ours(SH)](https://1drv.ms/u/s!AgY2evoYo6Fggrkt3viYwx2nNNSiPA?e=0hr90J) | 2 | 24.8 | 14.69 | 0.5325 | 0.6797 |
| [ours(SH+D)](https://1drv.ms/u/s!AgY2evoYo6Fggr4nRezW9tczj5T_gQ?e=cajXwk) | 8 | 60.8 | 30.53 | 0.8546 | 0.3369 |
| [ours(SH+D)](https://1drv.ms/u/s!AgY2evoYo6FggrsL-0Q_IqyQs6Uplw?e=mFAQUa) | 4 | 34.8 | 23.07 | 0.7238 | 0.5166 |
| [ours(SH+D)](https://1drv.ms/u/s!AgY2evoYo6Fggr5KBVxXMuz4FvkcAg?e=EDiqvd)| 2 | 21.8 | 14.65 | 0.5280 | 0.9981 |
| [clipping](https://1drv.ms/u/s!AgY2evoYo6Fggr9XzCHDzKU9sibGKg?e=V1LNUJ) | 8 | 62.6 | 17.78 | 0.7227 | 0.4879 |
| [clipping](https://1drv.ms/u/s!AgY2evoYo6FggrpkAHDd6gNvTOjchA?e=FLXhmh) | 4 | 37.4 | 17.72 | 0.6575 | 0.5502 |
| [clipping](https://1drv.ms/u/s!AgY2evoYo6FggsAdvtQjpBnB2jc8cA?e=nByLQY) | 2 | 24.4 | 16.66 | 0.6476 | 0.5881 |

We observed that our method shows the best performance and has reasonable memory requirement. 

## Experiment 3) Progressive Scaling

We found an interesting observation, the **progressive scaling** of Plenoxels strongly affects the memory footprint. We thus explore two progressive scaling methods: “weight” and “threshold”. According to [Plenoxels](https://arxiv.org/abs/2112.05131), “weight” filters out voxels by casting rays from training views and “sigma” directly prunes voxels whose densities are below the desired threshold. For more details, please refer to the supplementary materials of Plenoxels. For each method, we change the threshold of each scaling method to find a good adjustment between memory usage and rendering quality.

| Quantization Method | Threshold | Memory | PSNR | SSIM | LPIPS |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| [sigma](https://1drv.ms/u/s!AgY2evoYo6FggsEGqiwAb5fe4cLkbQ?e=Do1xfF) | 5 | 69.0 | 30.66 | 0.8554 | 0.3352 |
| [sigma](https://1drv.ms/u/s!AgY2evoYo6FggsE6bw7iWlF1E4zzKw?e=71rLxy) | 10 | 66.4 | 30.79 | 0.8550 | 0.3354 |
| [sigma](https://1drv.ms/u/s!AgY2evoYo6Fggr0Z9cV-P4IiQ67lsQ?e=pJtFef) | 20 | 63.2 | 30.81 | 0.8551 | 0.3363 |
| [sigma](https://1drv.ms/u/s!AgY2evoYo6FggsFotXycRremRu1fjw?e=xCVgfu) | 100 | 49.8 | 30.47 | 0.8547 | 0.3537 |
| [weight](https://1drv.ms/u/s!AgY2evoYo6FggsIWdOEpFvFe54QMiQ?e=jr63Dp) | 1.28 | 75.8 | 30.81 | 0.8557 | 0.3333 |
| [weight](https://1drv.ms/u/s!AgY2evoYo6FggsJmBsfWLWMYC8ih7A?e=O9rtOz) | 2.56 | 73.0 | 30.71 | 0.8560 | 0.3335 |

Models progressively scaled by the “sigma” method generally requires less memory than models progressively scaled with the “weight” method, although gaps between the two approaches are negligible. We select the “sigma” method as our progressive scaling method with a sigma threshold 20, which has shown the best trade-off between memory footprint and rendering quality. 
