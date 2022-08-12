---
layout: default
title: 3D Semantic Segmentation
parent: Benchmark
nav_order: 2
---

# 3D Semantic Segmentation

## Track 1: 3D semantic segmentation (No Feat)

In this track, the models are trained only using voxel coordinates on PeRFception-CO3D. In other words, using spherical harmonic coefficients and density values are prohibited. We sort the models based on the mIoU score. Models are trained for three times. We report mean and std for each model. 


|Model| mIoU | mAcc | Checkpoint | Code |
|:-:|:-:|:-:|:-:|:-:|
| [Mink-ResNet34](http://openaccess.thecvf.com/content_CVPR_2019/papers/Choy_4D_Spatio-Temporal_ConvNets_Minkowski_Convolutional_Neural_Networks_CVPR_2019_paper.pdf) | 60.2 $$\pm$$ 2.3 | 70.1 $$\pm$$ 1.9 | [link]() | [link]() |
| [Mink-ResNet18](http://openaccess.thecvf.com/content_CVPR_2019/papers/Choy_4D_Spatio-Temporal_ConvNets_Minkowski_Convolutional_Neural_Networks_CVPR_2019_paper.pdf) | 59.5 $$\pm$$ 2.0 | 69.7 $$\pm$$ 1.7 | [link]() | [link]() |
| [Mink-ResNet14](http://openaccess.thecvf.com/content_CVPR_2019/papers/Choy_4D_Spatio-Temporal_ConvNets_Minkowski_Convolutional_Neural_Networks_CVPR_2019_paper.pdf) | 57.6 $$ \pm$$ 2.0 | 67.9 $$\pm$$ 1.7 | [link]() | [link]() |

## Track 2: 3D semantic segmentation (SH)

In this track, the models are trained with spherical harmonic coefficients and voxel coordinates on PeRFception-CO3D. We sort the models based on the mIoU score. Models are trained for three times. We report mean and std for each model. 


|Model| mIoU | mAcc | Checkpoint | Code |
|:-:|:-:|:-:|:-:|:-:|
| [Mink-ResNet34](http://openaccess.thecvf.com/content_CVPR_2019/papers/Choy_4D_Spatio-Temporal_ConvNets_Minkowski_Convolutional_Neural_Networks_CVPR_2019_paper.pdf) | 62.5 $$\pm$$ 0.7 | 72.0 $$\pm$$ 0.6 |[link]() | [link]() |
| [Mink-ResNet18](http://openaccess.thecvf.com/content_CVPR_2019/papers/Choy_4D_Spatio-Temporal_ConvNets_Minkowski_Convolutional_Neural_Networks_CVPR_2019_paper.pdf) | 61.8 $$\pm$$ 0.4 | 71.7 $$\pm$$ 0.3 | [link]() | [link]() |
| [Mink-ResNet14](http://openaccess.thecvf.com/content_CVPR_2019/papers/Choy_4D_Spatio-Temporal_ConvNets_Minkowski_Convolutional_Neural_Networks_CVPR_2019_paper.pdf) | 60.2 $$\pm$$ 0.2 | 69.9 $$\pm$$ 0.4 | [link]() | [link]() |

## Track 3: 3D semantic segmentation (D)

In this track, the models are trained with density values and voxel coordinates on PeRFception-CO3D. We sort the models based on the mIoU score. Models are trained for three times. We report mean and std for each model.  


|Model| Acc@1 | Acc@5 | Checkpoint | Code |
|:-:|:-:|:-:|:-:|:-:|
| [Mink-ResNet34](http://openaccess.thecvf.com/content_CVPR_2019/papers/Choy_4D_Spatio-Temporal_ConvNets_Minkowski_Convolutional_Neural_Networks_CVPR_2019_paper.pdf) | 62.2 $$\pm$$ 0.9 | 72.1 $$\pm$$ 0.5 | [link]() | [link]() |
| [Mink-ResNet18](http://openaccess.thecvf.com/content_CVPR_2019/papers/Choy_4D_Spatio-Temporal_ConvNets_Minkowski_Convolutional_Neural_Networks_CVPR_2019_paper.pdf) | 61.5 $$\pm$$ 0.6 | 71.7 $$\pm$$ 0.4 | [link]() | [link]() |
| [Mink-ResNet14](http://openaccess.thecvf.com/content_CVPR_2019/papers/Choy_4D_Spatio-Temporal_ConvNets_Minkowski_Convolutional_Neural_Networks_CVPR_2019_paper.pdf) | 60.4 $$\pm$$ 0.1 | 70.4 $$\pm$$ 0.0| [link]() | [link]() |

## Track 4: 3D semantic segmentation (SH + D)

In this track, spherical harmonic coefficients, density, and voxel coordinates are available. We sort the models based on the mIoU score. Models are trained for three times. We report mean and std for each model. 


|Model| Acc@1 | Acc@5 | Checkpoint | Code |
|:-:|:-:|:-:|:-:|:-:|
| [Mink-ResNet34](http://openaccess.thecvf.com/content_CVPR_2019/papers/Choy_4D_Spatio-Temporal_ConvNets_Minkowski_Convolutional_Neural_Networks_CVPR_2019_paper.pdf) | 62.5 $$\pm$$ 0.7 | 72.2 $$\pm$$ 0.4 | [link]() | [link]() |
| [Mink-ResNet18](http://openaccess.thecvf.com/content_CVPR_2019/papers/Choy_4D_Spatio-Temporal_ConvNets_Minkowski_Convolutional_Neural_Networks_CVPR_2019_paper.pdf) | 61.7 $$\pm$$ 0.5 | 71.4 $$\pm$$ 0.4 | [link]() | [link]() |
| [Mink-ResNet14](http://openaccess.thecvf.com/content_CVPR_2019/papers/Choy_4D_Spatio-Temporal_ConvNets_Minkowski_Convolutional_Neural_Networks_CVPR_2019_paper.pdf) | 60.3 $$\pm$$ 0.1 | 70.0 $$\pm$$ 0.3 | [link]() | [link]()  |