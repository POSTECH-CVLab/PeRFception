# JaxNeRF

Here, we explain several details about the implementation for convenient usage. Our implementation is fully inspired by the official jax-implementation of NeRF(https://github.com/google-research/google-research/tree/master/jaxnerf). We observed several differences with the original TF-implementation of NeRF(https://github.com/bmild/nerf). 

In addition, we have implemented the heavy model, dubbed JaxNeRF+, to reproduce the result. Since many of researchers do not have large GPUs or TPUs, our code facilitates multi-node training. 

## Properties

:white_check_mark: DDP support!

:white_check_mark: Large Model (JaxNeRF+)

## Differences
This section compares the JaxNeRF and the original NeRF. 

- Precropping stage
    - The original NeRF crops center for early steps when training on blender scenes. Authors explained this is because of the early divergence of NeRF. 
    - Instead of the pre-cropping, the JaxNeRF handles this issue by introducing a delayed learning rate decay strategy. We found several scenes in the blender dataset are sensitive to seeds. Thus, we fixed the delay hyperparameters over all scenes in the blender dataset. We followed details from the original JaxNeRF.
- Batch Sampling
    - When selecting rays among all images(LLFF), JaxNeRF replaces selected rays all the time; however, the TFNeRF does not replace the selected rays. 
    - When selecting rays from a randonly selected single image(Blender), ray selection strategies of JaxNeRF and TFNeRF are implemented identically.

## News

- 21.12.14: The first version has been released. 

## Reproduced Performance

### Blender (NeRF-Synthetic)

#### Base Model (JaxNeRF)
| | Chair | Drums | Ficus | Hotdog | Lego | Materials | Mic | Ship |
|--- |---|---|---|---|---|---|---|---|
| PSNR (Test) | | | | | | | | |
| SSIM (Test) | | | | | | | | |
| LPIPS (Test) | | | | | | | | |
| PSNR (Overall) | | | | | | | | |
| SSIM (Overall) | | | | | | | | |
| LPIPS (Overall) | | | | | | | | |

#### Large Model (JaxNeRF+)
| | Chair | Drums | Ficus | Hotdog | Lego | Materials | Mic | Ship |
|--- |---|---|---|---|---|---|---|---|
| PSNR (Test) | | | | | | | | |
| SSIM (Test) | | | | | | | | |
| LPIPS (Test) | | | | | | | | |
| PSNR (Overall) | | | | | | | | |
| SSIM (Overall) | | | | | | | | |
| LPIPS (Overall) | | | | | | | | |

### LLFF (NeRF-Real)

#### Base Model (JaxNeRF)
| | Fern | Flower | Fortress | Horns | Leaves | Orchids | Room | Trex |
|--- |---|---|---|---|---|---|---|---|
| PSNR (Test) | | | | | | | | |
| SSIM (Test) | | | | | | | | |
| LPIPS (Test) | | | | | | | | |
| PSNR (Overall) | | | | | | | | |
| SSIM (Overall) | | | | | | | | |
| LPIPS (Overall) | | | | | | | | |

#### Large Model (JaxNeRF+)
| | Fern | Flower | Fortress | Horns | Leaves | Orchids | Room | Trex |
|--- |---|---|---|---|---|---|---|---|
| PSNR (Test) | | | | | | | | |
| SSIM (Test) | | | | | | | | |
| LPIPS (Test) | | | | | | | | |
| PSNR (Overall) | | | | | | | | |
| SSIM (Overall) | | | | | | | | |
| LPIPS (Overall) | | | | | | | | |

## Running the code
```
# Prerequisite
mkdir data && cd data

## LLFF
ln -s [path to nerf_llff_data] llff

## Blender
ln -s [path to nerf_blender_data] blender
```


```
# Blender - (Base, Large)

python3 -m run --config configs/jaxnerf/blender.yaml --expname [scene] --datadir data/blender/[scene] --train --eval
python3 -m run --config configs/jaxnerf/blender_large.yaml --expname [scene]_large --datadir data/blender/[scene] --train --eval

# LLFF - (Base, Large)

python3 -m run --config configs/jaxnerf/llff.yaml --expname [scene] --datadir data/llff/[scene] --train --eval
python3 -m run --config configs/jaxnerf/llff_large.yaml --expname [scene]_large --datadir data/llff/[scene] --train --eval
```
