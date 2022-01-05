# Plenoxel

Plenoxel has shown dramatic improvements in training time by directly optimizing sparse voxel grids. The primary version has been released in [Link](https://github.com/sxyu/svox2), however there are several excessive parts that are not used in the code, resulting in difficulties when modifying the code. Here, we reorganized the code so that the proposed model can be conveninently compared with other baselines. 

## Properties

:white_check_mark: Single GPU only!
:white_check_mark: nearly 30mins for training

## Setup
Before running the code, users should install an additional CUDA package.
```
export LD_LIBRARY_PATH=[path to environment]/lib/python[version]/site-packages/torch/lib/:$LD_LIBRARY_PATH
# Example: export LD_LIBRARY_PATH=/home/abcd/anaconda3/envs/env_name/lib/python3.7/site-packages/torch/lib/:$LD_LIBRARY_PATH
pip install .
```

## Reproduced Performance

### Blender

| | Chair | Drums | Ficus | Hotdog | Lego | Materials | Mic | Ship |
|--- |---|---|---|---|---|---|---|---|
| PSNR (Test) | 32.51 | 24.95 | 31.24 | 35.00 | 32.15 | 28.37 | 31.59 | 28.34 |
| SSIM (Test) | 0.9659 | 0.9266 | 0.9718 | 0.9761 | 0.9660 | 0.9445 | 0.9788 | 0.8780 |
| LPIPS (Test) | 0.03964 | 0.07283 | 0.02956 | 0.04052 | 0.04026 | 0.06316 | 0.02144 | 0.1473 |
| PSNR (All) | 33.16 | 26.86 | 32.85 | 36.28 | 33.60 | 31.09 | 32.26 | 30.22 |
| SSIM (All) | 0.9693 | 0.9446 | 0.9809 | 0.9806 | 0.9742 | 0.9675 | 0.9818 | 0.9023 |
| LPIPS (All) | 0.03502 | 0.06379 | 0.02185 | 0.03321 | 0.03282 | 0.04526 | 0.01841 | 0.1274 |

### LLFF (NeRF-Real)

| | Fern | Flower | Fortress | Horns | Leaves | Orchids | Room | Trex |
|--- |---|---|---|---|---|---|---|---|
| PSNR (Test) | 25.50 | 27.82 | 31.08 | 27.47 | 21.46 | 20.51 | 30.33 | 26.26 |
| SSIM (Test) | 0.8282 | 0.8642 | 0.8809 | 0.8504 | 0.7589 | 0.6838 | 0.9388 | 0.8846 |
| LPIPS (Test) | 0.2270 | 0.1819 | 0.1853 | 0.2385 | 0.1992 | 0.2642 | 0.1898 | 0.2464 |
| PSNR (All) | 27.59 | 30.85 | 32.31 | 28.54 | 23.84 | 24.19 | 33.51 | 28.45 |
| SSIM (All) | 0.8755 | 0.9117 | 0.8963 | 0.8670 | 0.303 | 0.7932 | 0.9555 | 0.9192 |
| LPIPS (All) | 0.1897 | 0.1375 | 0.1712 | 0.2214 | 0.1604 | 0.2099 | 0.1670 | 0.1996 |


### Tanks and Temples

| | M60 | Train | Truck | Playground | 
|--- |---|---|---|---|
| PSNR (Test) | 25.50 | 27.82 | 21.23 | 27.47 |
| SSIM (Test) | 0.8282 | 0.8642 | 0.6938 | 0.8504 | 
| LPIPS (Test) | 0.2270 | 0.1819 | 0.4456 | 0.2385 | 
| PSNR (All) | 27.59 | 30.85 | 23.34 | 28.54 |
| SSIM (All) | 0.8755 | 0.9117 | 0.7472 | 0.8670 |
| LPIPS (All) | 0.1897 | 0.1375 | 0.3743 | 0.2214 | 