# JaxNeRF

Here, we explain several details about the implementation for convenient usage. Our implementation is fully inspired by the official jax-implementation of NeRF(https://github.com/google-research/google-research/tree/master/jaxnerf). We observed several differences with the original TF-implementation of NeRF(https://github.com/bmild/nerf). 

In addition, we have implemented the heavy model, dubbed JaxNeRF+, to reproduce the result. Since many of researchers do not have large GPUs or TPUs, our code facilitates multi-node training. 

## Properties

:white_check_mark: DDP support!

:white_check_mark: Large Model (JaxNeRF+)

## Differences
This section compares the JaxNeRF and the original NeRF. In addition, we also provide additional details that we have changed in this implementation and their corrsponding reasons.

- Precropping stage
    - The original NeRF crops center for early steps when training on blender scenes. Authors explained this is because of the early divergence of NeRF. 
    - Instead of the pre-cropping, the JaxNeRF handles this issue by introducing a delayed learning rate decay strategy. We found several scenes in the blender dataset are sensitive to seeds. Thus, we fixed the delay hyperparameters over all scenes in the blender dataset. We followed details from the original JaxNeRF.
- Batch Sampling
    - When selecting rays among all images(LLFF), JaxNeRF replaces selected rays all the time; however, the TFNeRF does not replace the selected rays. 
    - When selecting rays from a randonly selected single image(Blender), ray selection strategies of JaxNeRF and TFNeRF are implemented identically.
- Validation Phase
    - Most of the NeRF implementations utilize the validation stage that was done with a randomly selected image rahter than the full validation set. 
    However, it is more convincible to check the quantitative qualities using the whole images in the validation set. 
    Our implementation corrected the validation phase to role as "validation". In other words, our final model is selected as the best model, evaluated from 
    the full validation set.

## News

- 21.12.14: The first version has been released. 

## Reproduced Performance

The results below are from the setting in our repository. Due to limited resources, we only trained 200K iterations. JaxNeRF is trained with 1M iterations. 
For any performance comparison, please set the `max_steps` to be equal in experiments. Please refer to the scripts in `scripts/jaxnerf_torch`

We provide the pretrained checkpoints, per-scene qualities, and mean qualities for each scene in the [[link]](https://drive.google.com/file/d/1bBDwLyxQBe4JE_0zsYxtb_3sXnpkGxM7/view?usp=sharing) 

### Blender (NeRF-Synthetic)

Although we only ran 200K iterations(1M in the official JaxNeRF+), our implementation achieves better quality then JaxNeRF when training a large model.

#### Large Model (JaxNeRF+)
| | Chair | Drums | Ficus | Hotdog | Lego | Materials | Mic | Ship |
|--- |---|---|---|---|---|---|---|---|
| PSNR (Test) | 35.26 | 26.17 | 34.45 | 37.73 | 36.35 | 30.52 | 35.85 | 30.95 |
| SSIM (Test) | 0.9817 | 0.9417 | 0.9842 | 0.9818 | 0.9803 | 0.9582 | 0.9892 | 0.8878 |
| LPIPS (Test) | 0.02711 | 0.06676 | 0.01892 | 0.03351 | 0.02273 | 0.05283 | 0.01407 | 0.1522 |
| PSNR (All) | 36.21 | 29.01 | 36.32 | 38.77 | 38.09 | 33.05 | 37.13 | 31.70 |
| SSIM (All) | 0.9845 | 0.9548 | 0.9887 | 0.9830 | 0.9833 | 0.9696 | 0.9915 | 0.8795 |
| LPIPS (All) | 0.02344 | 0.06243 | 0.01603 | 0.03284 | 0.01997 | 0.04582 | 0.01165 | 0.1630 |

#### Base Model (JaxNeRF)
| | Chair | Drums | Ficus | Hotdog | Lego | Materials | Mic | Ship |
|--- |---|---|---|---|---|---|---|---|
| PSNR (Test) | 32.77 | 25.16 | 31.02 | 36.18 | 32.35 | 29.37 | 32.83 | 28.45 |
| SSIM (Test) | 0.9649 | 0.9254 | 0.9671 | 0.9733 | 0.9567 | 0.9467 | 0.9790 | 0.8515 |
| LPIPS (Test) | 0.05180 | 0.09268 | 0.03914 | 0.04926 | 0.05777 | 0.06493 | 0.02733 | 0.2066|
| PSNR (All) | 33.39 | 26.73 | 32.11 | 36.90 | 33.46 | 30.92 | 33.23 | 29.02 |
| SSIM (All) | 0.9681 | 0.9352 | 0.9731 | 0.9739 | 0.9607 | 0.9571 | 0.9801 | 0.8393 |
| LPIPS (All) | 0.04768 | 0.08990 | 0.03541 | 0.05438 | 0.05499 | 0.05857 | 0.02511 | 0.2199 |

### LLFF (NeRF-Real)

#### Large Model (JaxNeRF+)
| | Fern | Flower | Fortress | Horns | Leaves | Orchids | Room | Trex |
|--- |---|---|---|---|---|---|---|---|
| PSNR (Test) | 25.32 | 28.13 | 31.72 | 28.47 | 21.12 | 20.17 | 33.46 | 27.51 |
| SSIM (Test) | 0.8161 | 0.8601 | 0.9001 | 0.8761 | 0.7250 | 0.6520 | 0.9594 | 0.9126 |
| LPIPS (Test) | 0.2397 | 0.1734 | 0.1410 | 0.1948 | 0.2615 | 0.3030 | 0.1418 | 0.2044 |
| PSNR (All) | 27.49 | 31.83 | 33.55 | 30.02 | 23.11 | 23.23 | 38.20 | 30.03 |
| SSIM (All) | 0.8555 | 0.9084 | 0.9183 | 0.8891 | 0.7828 | 0.7434 | 0.9724 | 0.9315 |
| LPIPS (All) | 0.2145 | 0.1348 | 0.1275 | 0.1835 | 0.2316 | 0.2661| 0.1243 | 0.1730 |

#### Base Model (JaxNeRF)
| | Fern | Flower | Fortress | Horns | Leaves | Orchids | Room | Trex |
|--- |---|---|---|---|---|---|---|---|
| PSNR (Test) | 24.83 | 27.06 | 30.89 | 26.79 | 20.75 | 20.24 | 31.94 | 26.08 |
| SSIM (Test) | 0.7710 | 0.8110 | 0.8643 | 0.7990 | 0.6690 | 0.6296 | 0.9436 | 0.8614 |
| LPIPS (Test) | 0.3169 | 0.2459 | 0.2047 | 0.3171 | 0.3409 | 0.3385 | 0.1937 | 0.2804 |
| PSNR (All) | 25.74 | 29.07 | 31.68 | 27.58 | 21.77 | 22.43 | 34.84 | 27.54 |
| SSIM (All) | 0.7900 | 0.8507 | 0.8737 | 0.8052 | 0.7049 | 0.6948 | 0.9551 | 0.8824|
| LPIPS (All) | 0.3029 | 0.2112 | 0.1958 | 0.3107 | 0.3213 | 0.3130 | 0.1778 | 0.2536 |


## Running the code
```
# Prerequisite
mkdir data && cd data

## LLFF
ln -s [path to nerf_llff_data] llff

## Blender
ln -s [path to nerf_blender_data] blender
```

All the scripts are provided in `scripts/jaxnerf_torch/[scripts].sh`