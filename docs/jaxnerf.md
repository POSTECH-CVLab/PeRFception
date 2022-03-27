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
However, one of the shortcomings of JaxNeRF+ is its unstable training process on the blender dataset. As authors mentioned in the official repository, 
the performance strongly depends on the seed we use. Consider using different learning rate warmup configs when the model fails to be trained. 

#### Large Model (JaxNeRF+)
| | Chair | Drums | Ficus* | Hotdog | Lego | Materials | Mic* | Ship |
|--- |---|---|---|---|---|---|---|---|
| PSNR (Test) | 35.23 | 27.12 | 34.45 | 37.46 | 37.26 | 31.08 | 35.85 | 31.88 |
| SSIM (Test) | 0.9823 | 0.9446 | 0.9842 | 0.9804 | 0.9816 | 0.9623 | 0.9892 | 0.8743 |
| LPIPS (Test) | 0.02398 | 0.07061 | 0.01892 | 0.03685 | 0.02082 | 0.05041 | 0.01407 | 0.1762 |

#### Base Model (JaxNeRF)
| | Chair | Drums | Ficus | Hotdog | Lego | Materials | Mic | Ship |
|--- |---|---|---|---|---|---|---|---|
| PSNR (Test) | 32.79 | 25.17 | 31.01 | 36.13 | 32.52 | 29.42 | 33.01 | 28.43 |
| SSIM (Test) | 0.9648 | 0.9254 | 0.9666 | 0.9727 | 0.9576 | 0.9477 | 0.9796 | 0.8509 |
| LPIPS (Test) | 0.05187 | 0.09157 | 0.03960 | 0.05104 | 0.05705 | 0.06397 | 0.02682 | 0.2053 |

### LLFF (NeRF-Real)

#### Large Model (JaxNeRF+)
| | Fern | Flower | Fortress | Horns | Leaves | Orchids | Room | Trex |
|--- |---|---|---|---|---|---|---|---|
| PSNR (Test) | 25.32 | 28.13 | 31.72 | 28.47 | 21.12 | 20.17 | 33.46 | 27.51 |
| SSIM (Test) | 0.8161 | 0.8601 | 0.9001 | 0.8761 | 0.7250 | 0.6520 | 0.9594 | 0.9126 |
| LPIPS (Test) | 0.2397 | 0.1734 | 0.1410 | 0.1948 | 0.2615 | 0.3030 | 0.1418 | 0.2044 |

#### Base Model (JaxNeRF)
| | Fern | Flower | Fortress | Horns | Leaves | Orchids | Room | Trex |
|--- |---|---|---|---|---|---|---|---|
| PSNR (Test) | 24.83 | 27.06 | 30.89 | 26.79 | 20.75 | 20.24 | 31.94 | 26.08 |
| SSIM (Test) | 0.7710 | 0.8110 | 0.8643 | 0.7990 | 0.6690 | 0.6296 | 0.9436 | 0.8614 |
| LPIPS (Test) | 0.3169 | 0.2459 | 0.2047 | 0.3171 | 0.3409 | 0.3385 | 0.1937 | 0.2804 |


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