---
layout: default
title: 3D Classification
parent: Benchmark
nav_order: 2
---

# 3D Classification

## Track 1: 3D classification (No Feat)

In this track, the models are trained only using voxel coordinates on PeRFception-CO3D. In other words, using spherical harmonic coefficients and density values are prohibited. We sort the models based on the Acc@1 score. Models are trained for three times. We report mean and std for each model. 


|Model| Acc@1 | Acc@5 | Checkpoint | Code |
|:-:|:-:|:-:|:-:|:-:|
| [Mink-ResNet101](http://openaccess.thecvf.com/content_CVPR_2019/papers/Choy_4D_Spatio-Temporal_ConvNets_Minkowski_Convolutional_Neural_Networks_CVPR_2019_paper.pdf) | 66.21 $$\pm$$ 0.93 | 86.50 $$\pm$$ 0.09 | [link](https://1drv.ms/u/s!AgY2evoYo6FggrFeAn3wJhvNQk28qg?e=6sDRBc) | [link](https://github.com/POSTECH-CVLab/NeRF-Downstream) |
| [Mink-ResNet50](http://openaccess.thecvf.com/content_CVPR_2019/papers/Choy_4D_Spatio-Temporal_ConvNets_Minkowski_Convolutional_Neural_Networks_CVPR_2019_paper.pdf) | 65.25 $$\pm$$ 0.75 | 85.71 $$\pm$$ 0.64 | [link](https://1drv.ms/u/s!AgY2evoYo6FggrFW9e2K8H40QvRusA?e=F7crQc) | [link](https://github.com/POSTECH-CVLab/NeRF-Downstream) |
| [Mink-ResNet34](http://openaccess.thecvf.com/content_CVPR_2019/papers/Choy_4D_Spatio-Temporal_ConvNets_Minkowski_Convolutional_Neural_Networks_CVPR_2019_paper.pdf) | 64.55 $$\pm$$ 0.84 | 84.26 $$\pm$$ 0.47 | [link](https://1drv.ms/u/s!AgY2evoYo6FggrFNFM3adn80njGTLw?e=8Engvz) | [link](https://github.com/POSTECH-CVLab/NeRF-Downstream) |
| [Mink-ResNet18](http://openaccess.thecvf.com/content_CVPR_2019/papers/Choy_4D_Spatio-Temporal_ConvNets_Minkowski_Convolutional_Neural_Networks_CVPR_2019_paper.pdf) | 63.85 $$\pm$$ 0.33 | 84.47 $$\pm$$ 0.54 | [link](https://1drv.ms/u/s!AgY2evoYo6FggrFEfSJ9_xiTSEqeHg?e=FStR0b) | [link](https://github.com/POSTECH-CVLab/NeRF-Downstream) |
| [Mink-ResNet14](http://openaccess.thecvf.com/content_CVPR_2019/papers/Choy_4D_Spatio-Temporal_ConvNets_Minkowski_Convolutional_Neural_Networks_CVPR_2019_paper.pdf) | 59.36 $$\pm$$ 0.30 | 81.30 $$\pm$$ 0.87 | [link](https://1drv.ms/u/s!AgY2evoYo6FggrE_nE1hJ_6dC_tYSg?e=4jjHVm) | [link](https://github.com/POSTECH-CVLab/NeRF-Downstream) |

## Track 2: 3D classification (SH)

In this track, the models are trained with spherical harmonic coefficients and voxel coordinates on PeRFception-CO3D. We sort the models based on the Acc@1 score. Models are trained for three times. We report mean and std for each model. 


|Model| Acc@1 | Acc@5 | Checkpoint | Code |
|:-:|:-:|:-:|:-:|:-:|
| [Mink-ResNet101](http://openaccess.thecvf.com/content_CVPR_2019/papers/Choy_4D_Spatio-Temporal_ConvNets_Minkowski_Convolutional_Neural_Networks_CVPR_2019_paper.pdf) | 78.04 $$\pm$$ 0.58 | 89.60 $$\pm$$ 0.36 | [link](https://1drv.ms/u/s!AgY2evoYo6FggrIfn0RqH6-DDnkPCA?e=U1zMYB) | [link](https://github.com/POSTECH-CVLab/NeRF-Downstream) |
| [Mink-ResNet50](http://openaccess.thecvf.com/content_CVPR_2019/papers/Choy_4D_Spatio-Temporal_ConvNets_Minkowski_Convolutional_Neural_Networks_CVPR_2019_paper.pdf) | 77.59 $$\pm$$ 0.17 | 90.98 $$\pm$$ 0.40 | [link](https://1drv.ms/u/s!AgY2evoYo6FggrIfn0RqH6-DDnkPCA?e=U1zMYB) | [link](https://github.com/POSTECH-CVLab/NeRF-Downstream) |
| [Mink-ResNet34](http://openaccess.thecvf.com/content_CVPR_2019/papers/Choy_4D_Spatio-Temporal_ConvNets_Minkowski_Convolutional_Neural_Networks_CVPR_2019_paper.pdf) | 76.51 $$\pm$$ 0.61 | 91.79 $$\pm$$ 0.49 | [link](https://1drv.ms/u/s!AgY2evoYo6FggrIaBh9bxxKLl1J3vA?e=T779tB) | [link](https://github.com/POSTECH-CVLab/NeRF-Downstream) |
| [Mink-ResNet18](http://openaccess.thecvf.com/content_CVPR_2019/papers/Choy_4D_Spatio-Temporal_ConvNets_Minkowski_Convolutional_Neural_Networks_CVPR_2019_paper.pdf) | 75.58 $$\pm$$ 0.37 | 92.73 $$\pm$$ 0.21 | [link](https://1drv.ms/u/s!AgY2evoYo6FggrIU9vdqckhVQmJa5Q?e=uW5g8z) | [link](https://github.com/POSTECH-CVLab/NeRF-Downstream) |
| [Mink-ResNet14](http://openaccess.thecvf.com/content_CVPR_2019/papers/Choy_4D_Spatio-Temporal_ConvNets_Minkowski_Convolutional_Neural_Networks_CVPR_2019_paper.pdf) | 71.87 $$\pm$$ 0.61 | 93.24 $$\pm$$ 0.49 | [link](https://1drv.ms/u/s!AgY2evoYo6FggrIQpmXLoJarWB5Zlw?e=uidtdT) | [link](https://github.com/POSTECH-CVLab/NeRF-Downstream) |


## Track 3: 3D classification (D)

In this track, the models are trained with density values and voxel coordinates on PeRFception-CO3D. We sort the models based on the Acc@1 score. Models are trained for three times. We report mean and std for each model. 


|Model| Acc@1 | Acc@5 | Checkpoint | Code |
|:-:|:-:|:-:|:-:|:-:|
| [Mink-ResNet101](http://openaccess.thecvf.com/content_CVPR_2019/papers/Choy_4D_Spatio-Temporal_ConvNets_Minkowski_Convolutional_Neural_Networks_CVPR_2019_paper.pdf) | 77.27 $$\pm$$ 0.61 | 92.68 $$\pm$$ 0.34 | [link](https://1drv.ms/u/s!AgY2evoYo6FggrIGRCnKoPdvOrAqrA?e=XgIWkw) | [link](https://github.com/POSTECH-CVLab/NeRF-Downstream) |
| [Mink-ResNet50](http://openaccess.thecvf.com/content_CVPR_2019/papers/Choy_4D_Spatio-Temporal_ConvNets_Minkowski_Convolutional_Neural_Networks_CVPR_2019_paper.pdf) | 76.42 $$\pm$$ 0.19 | 92.91 $$\pm$$ 0.24 | [link](https://1drv.ms/u/s!AgY2evoYo6FggrF5TWk8H8Vx7Zf43w?e=vBAK5g) | [link](https://github.com/POSTECH-CVLab/NeRF-Downstream) |
| [Mink-ResNet34](http://openaccess.thecvf.com/content_CVPR_2019/papers/Choy_4D_Spatio-Temporal_ConvNets_Minkowski_Convolutional_Neural_Networks_CVPR_2019_paper.pdf) | 76.38 $$\pm$$ 0.34 | 91.79 $$\pm$$ 0.61 | [link](https://1drv.ms/u/s!AgY2evoYo6FggrF5TWk8H8Vx7Zf43w?e=vBAK5g) | [link](https://github.com/POSTECH-CVLab/NeRF-Downstream) |
| [Mink-ResNet18](http://openaccess.thecvf.com/content_CVPR_2019/papers/Choy_4D_Spatio-Temporal_ConvNets_Minkowski_Convolutional_Neural_Networks_CVPR_2019_paper.pdf) | 75.18 $$\pm$$ 0.70 | 91.10 $$\pm$$ 0.24 | [link](https://1drv.ms/u/s!AgY2evoYo6FggrF0POYoofyKHkBH7A?e=WovA1p) | [link](https://github.com/POSTECH-CVLab/NeRF-Downstream) |
| [Mink-ResNet14](http://openaccess.thecvf.com/content_CVPR_2019/papers/Choy_4D_Spatio-Temporal_ConvNets_Minkowski_Convolutional_Neural_Networks_CVPR_2019_paper.pdf) | 72.44 $$\pm$$ 0.29 | 90.04 $$\pm$$ 0.13 | [link](https://1drv.ms/u/s!AgY2evoYo6FggrFuhJtJo7pR8bmp9g?e=NYyBG0) | [link](https://github.com/POSTECH-CVLab/NeRF-Downstream) |


## Track 4: 3D classification (SH + D)

In this track, spherical harmonic coefficients, density, and voxel coordinates are available. We sort the models based on the Acc@1 score. Models are trained for three times. We report mean and std for each model. 


|Model| Acc@1 | Acc@5 | Checkpoint | Code |
|:-:|:-:|:-:|:-:|:-:|
| [Mink-ResNet50](http://openaccess.thecvf.com/content_CVPR_2019/papers/Choy_4D_Spatio-Temporal_ConvNets_Minkowski_Convolutional_Neural_Networks_CVPR_2019_paper.pdf) | 77.53 $$\pm$$ 0.27 | 92.93 $$\pm$$ 0.54 | [link](https://1drv.ms/u/s!AgY2evoYo6FggrI9tKNKwuoWVupqRQ?e=PNInVr) | [link](https://github.com/POSTECH-CVLab/NeRF-Downstream) |
| [Mink-ResNet101](http://openaccess.thecvf.com/content_CVPR_2019/papers/Choy_4D_Spatio-Temporal_ConvNets_Minkowski_Convolutional_Neural_Networks_CVPR_2019_paper.pdf) | 77.19 $$\pm$$ 0.89 | 93.09 $$\pm$$ 0.21 | [link](https://1drv.ms/u/s!AgY2evoYo6FggrJCWAn8oilyJ7QT2A?e=LATygo) | [link](https://github.com/POSTECH-CVLab/NeRF-Downstream) |
| [Mink-ResNet34](http://openaccess.thecvf.com/content_CVPR_2019/papers/Choy_4D_Spatio-Temporal_ConvNets_Minkowski_Convolutional_Neural_Networks_CVPR_2019_paper.pdf) | 76.50 $$\pm$$ 0.03 | 91.98 $$\pm$$ 0.08 | [link](https://1drv.ms/u/s!AgY2evoYo6FggrIu1uGMyJ7RbNbgrw?e=vRr5UD) | [link](https://github.com/POSTECH-CVLab/NeRF-Downstream) |
| [Mink-ResNet18](http://openaccess.thecvf.com/content_CVPR_2019/papers/Choy_4D_Spatio-Temporal_ConvNets_Minkowski_Convolutional_Neural_Networks_CVPR_2019_paper.pdf) | 75.72 $$\pm$$ 0.25 | 91.54 $$\pm$$ 0.21 | [link](https://1drv.ms/u/s!AgY2evoYo6FggrIu1uGMyJ7RbNbgrw?e=vRr5UD) | [link](https://github.com/POSTECH-CVLab/NeRF-Downstream) |
| [Mink-ResNet14](http://openaccess.thecvf.com/content_CVPR_2019/papers/Choy_4D_Spatio-Temporal_ConvNets_Minkowski_Convolutional_Neural_Networks_CVPR_2019_paper.pdf) | 72.92 $$\pm$$ 0.42 | 90.83 $$\pm$$ 0.03 | [link](https://1drv.ms/u/s!AgY2evoYo6FggrIu1uGMyJ7RbNbgrw?e=vRr5UD) | [link](https://github.com/POSTECH-CVLab/NeRF-Downstream) |