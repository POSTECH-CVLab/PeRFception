---
layout: default
title: 2D Classification
parent: Benchmark
nav_order: 1
---

# 2D Classification

## Track 1: In-domain classification (Scratch)

The model is trained and tested on PeRFception-Co3D dataset. We sort models with the primary metric Acc@1. For this track, the models trained from scratch is allowed. No external dataset is allowed. Acc@1 stands for top-1 accuracy, and Acc@5 stands for top-5 accuracy.

|Model| Acc@1 | Acc@5 | Checkpoints | Code |
|:-:|:-:|:-:|:-:|:-:|
| [ResNext101](https://openaccess.thecvf.com/content_cvpr_2017/papers/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.pdf) | 85.48 $$\pm$$ 0.06 | 96.26 $$\pm$$ 0.03 | [link]() | [link]()
| [WideResNet101](https://arxiv.org/pdf/1605.07146) | 85.30 $$\pm$$ 0.11 | 96.31 $$\pm$$ 0.10 | [link]() | [link]() |
| [ResNet152](https://arxiv.org/pdf/1512.03385) | 85.28 $$\pm$$ 0.02 | 96.39 $$\pm$$ 0.06 | [link]() | [link]() |
| [ResNet101](https://arxiv.org/pdf/1512.03385) | 85.11 $$\pm$$ 0.23 | 96.32 $$\pm$$ 0.12 | [link]() | [link]() |
| [WideResNet50](https://arxiv.org/pdf/1605.07146) | 84.68 $$\pm$$ 0.02 | 96.03 $$\pm$$ 0.03 | [link]() | [link]() |
| [ResNext50](https://openaccess.thecvf.com/content_cvpr_2017/papers/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.pdf) | 84.32 $$\pm$$ 0.18 | 95.92 $$\pm$$ 0.16 | [link]() | [link]() |
| [ResNet50](https://arxiv.org/pdf/1512.03385) | 83.77 $$\pm$$ 0.08 | 95.99 $$\pm$$ 0.08 | [link]() | [link]()
| [ResNet34](https://arxiv.org/pdf/1512.03385) | 83.61 $$\pm$$ 0.04 | 95.89 $$\pm$$ 0.06 | [link]() | [link]()
| [ResNet18](https://arxiv.org/pdf/1512.03385) | 82.05 $$\pm$$ 0.24 | 95.37 $$\pm$$ 0.05 | [link]() | [link]()

## Track 2: In-domain classification (Pre-trained)

The model is trained on PeRFception-Co3D dataset and tested CO3D dataset. We sort models with the primary metric Acc@1. For this track, any pre-trained datasets are allowed. However, we prohibit external datasets that include any CO3D test images on training split.

|Model| Dataset | Acc@1 | Acc@5 | Checkpoints | Code |
|:-:|:-:|:-:|:-:|:-:|:-:|
| [ResNet152](https://arxiv.org/pdf/1512.03385) | [ImageNet](https://www.image-net.org/) | 88.73 $$\pm$$ 0.15 | 97.24 $$\pm$$ 0.08 | [link]() | [link]() |
| [WideResNet101](https://arxiv.org/pdf/1605.07146) | [ImageNet](https://www.image-net.org/) | 88.39 $$\pm$$ 0.07 | 96.31 $$\pm$$ 0.10 | [link]() | [link]() |
| [ResNext101](https://openaccess.thecvf.com/content_cvpr_2017/papers/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.pdf)  | [ImageNet](https://www.image-net.org/)  | 88.51 $$\pm$$ 0.16 | 96.93 $$\pm$$ 0.07 | [link]() | [link]()
| [ResNet101](https://arxiv.org/pdf/1512.03385) | [ImageNet](https://www.image-net.org/) | 88.32 $$\pm$$ 0.13 | 97.13 $$\pm$$ 0.05 | [link]() | [link]() |
| [WideResNet50](https://arxiv.org/pdf/1605.07146) | [ImageNet](https://www.image-net.org/)  | 87.75 $$\pm$$ 0.25 | 96.84 $$\pm$$ 0.09 | [link]() | [link]() |
| [ResNext50](https://openaccess.thecvf.com/content_cvpr_2017/papers/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.pdf) | [ImageNet](https://www.image-net.org/)  | 87.30 $$\pm$$ 0.16 | 96.66 $$\pm$$ 0.07 | [link]() | [link]() |
| [ResNet50](https://arxiv.org/pdf/1512.03385) | [ImageNet](https://www.image-net.org/)  | 87.30 $$\pm$$ 0.08 | 96.69 $$\pm$$ 0.08 | [link]() | [link]()
| [ResNet34](https://arxiv.org/pdf/1512.03385) | [ImageNet](https://www.image-net.org/)  | 86.25 $$\pm$$ 0.19 | 96.50 $$\pm$$ 0.09 | [link]() | [link]()
| [ResNet18](https://arxiv.org/pdf/1512.03385) | [ImageNet](https://www.image-net.org/)  | 84.97 $$\pm$$ 0.13 | 96.24 $$\pm$$ 0.09 | [link]() | [link]()


## Track 3: Out-domain classification
The model is trained on PeRFception-Co3D dataset and tested on Co3D dataset. Here, we evaluate Acc@1 on the original Co3D dataset. In addition, we 

|Model| Acc@1 | $$\mu_{perf} - \mu_{co3d}$$ | Checkpoints | Code |
|:-:|:-:|:-:|:-:|:-:|
| [ResNet152](https://arxiv.org/pdf/1512.03385) | 88.73 $$\pm$$ 0.15 | 97.24 $$\pm$$ 0.08 | [link]() | [link]() |
| [WideResNet101](https://arxiv.org/pdf/1605.07146) | 88.39 $$\pm$$ 0.07 | 96.31 $$\pm$$ 0.10 | [link]() | [link]() |
| [ResNext101](https://openaccess.thecvf.com/content_cvpr_2017/papers/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.pdf) | 88.51 $$\pm$$ 0.16 | 96.93 $$\pm$$ 0.07 | [link]() | [link]()
| [ResNet101](https://arxiv.org/pdf/1512.03385) | 88.32 $$\pm$$ 0.13 | 97.13 $$\pm$$ 0.05 | [link]() | [link]() |
| [WideResNet50](https://arxiv.org/pdf/1605.07146) | 87.75 $$\pm$$ 0.25 | 96.84 $$\pm$$ 0.09 | [link]() | [link]() |
| [ResNext50](https://openaccess.thecvf.com/content_cvpr_2017/papers/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.pdf) | 87.30 $$\pm$$ 0.16 | 96.66 $$\pm$$ 0.07 | [link]() | [link]() |
| [ResNet50](https://arxiv.org/pdf/1512.03385) | 87.30 $$\pm$$ 0.08 | 96.69 $$\pm$$ 0.08 | [link]() | [link]()
| [ResNet34](https://arxiv.org/pdf/1512.03385) | 86.25 $$\pm$$ 0.19 | 96.50 $$\pm$$ 0.09 | [link]() | [link]()
| [ResNet18](https://arxiv.org/pdf/1512.03385) | 84.97 $$\pm$$ 0.13 | 96.24 $$\pm$$ 0.09 | [link]() | [link]()



## Future Extension