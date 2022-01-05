# Plenoxel

Plenoxel has shown dramatic improvements in training time by directly optimizing sparse voxel grids. The primary version has been released in [Link](https://github.com/sxyu/svox2), however there are several excessive parts that are not used in the code, resulting in difficulties when modifying the code. Here, we reorganized the code so that the proposed model can be conveninently compared with other baselines. Again, we have not modified the code. 

## Properties

:white_check_mark: Single GPU only!
:white_check_mark: nearly 30mins for training

## Details
We found several interesting details in the official code, which are not explained in the paper. 
- For blender scenes, the scale of scene is reduced to 2/3. 
- There are two strategies to resample the sparse voxel grids: "sigma" and "weights". Refer to the code [Link](https://github.com/POSTECH-CVLab/NeRF-Factory/blob/c57a275af98e75b716e7f32a32b12b68a26b6d50/model/plenoxel_torch/sparse_grid.py#L991) and [Link](https://github.com/POSTECH-CVLab/NeRF-Factory/blob/c57a275af98e75b716e7f32a32b12b68a26b6d50/model/plenoxel_torch/sparse_grid.py#L1026) for details. 

## Setup
Before running the code, users should install an additional CUDA package.
```
export LD_LIBRARY_PATH=[path to environment]/lib/python[version]/site-packages/torch/lib/:$LD_LIBRARY_PATH
# Example: export LD_LIBRARY_PATH=/home/abcd/anaconda3/envs/env_name/lib/python3.7/site-packages/torch/lib/:$LD_LIBRARY_PATH
pip install .
```
Since the CUDA implementation only supports 32bits training, it is strongly recommended to train with 32 bits; several functions could cause side-effects. 

## Reproduced Performance

Due to limited storage of our Google Cloud, we do not provide the pretrained model for the plenoxel model(<= 7GB each). 
Instead, we provide scripts in `scripts/plenoxel_torch`, which can completely reproduce the performance reported below.
For a fair comparison with NeRF, we conducted experiments with batch size '4096' and the same number of iterations `200k`. 
In addition, we found the performance varies depending on the seed. 

### Blender (Fair Comparison with NeRF)

| | Chair | Drums | Ficus | Hotdog | Lego | Materials | Mic | Ship |
|--- |---|---|---|---|---|---|---|---|
| PSNR (Test) | 32.51 | 24.95 | 31.24 | 35.00 | 32.15 | 28.37 | 31.59 | 28.34 |
| SSIM (Test) | 0.9659 | 0.9266 | 0.9718 | 0.9761 | 0.9660 | 0.9445 | 0.9788 | 0.8780 |
| LPIPS (Test) | 0.03964 | 0.07283 | 0.02956 | 0.04052 | 0.04026 | 0.06316 | 0.02144 | 0.1473 |
| PSNR (All) | 33.16 | 26.86 | 32.85 | 36.28 | 33.60 | 31.09 | 32.26 | 30.22 |
| SSIM (All) | 0.9693 | 0.9446 | 0.9809 | 0.9806 | 0.9742 | 0.9675 | 0.9818 | 0.9023 |
| LPIPS (All) | 0.03502 | 0.06379 | 0.02185 | 0.03321 | 0.03282 | 0.04526 | 0.01841 | 0.1274 |

### Blender (The same setting with the paper)

| | Chair | Drums | Ficus | Hotdog | Lego | Materials | Mic | Ship |
|--- |---|---|---|---|---|---|---|---|
| PSNR (Test) | 32.56 | 24.94 | 31.24 | 35.02 | 32.14 | 28.34 | 31.67 | 28.31 |
| SSIM (Test) | 0.9663 | 0.9313 | 0.9715 | 0.9758 | 0.9655 | 0.9435 | 0.9791 | 0.8762 |
| LPIPS (Test) | 0.03923 | 0.07382 | 0.03049 | 0.04179 | 0.04072 | 0.06463 | 0.02188 | 0.1473 |
| PSNR (All) | 33.25 | 26.94 | 32.93 | 36.36 | 33.66 | 31.21 | 32.41 | 30.30 |
| SSIM (All) | 0.9699 | 0.9444 | 0.9811 | 0.9808 | 0.9744 | 0.9678 | 0.9824 | 0.9039 |
| LPIPS (All) | 0.03412 | 0.06401 | 0.02225 | 0.03334 | 0.03259 | 0.04523 | 0.01842 | 0.1241 |

### LLFF (NeRF-Real) (Fair Comparison with NeRF)

| | Fern | Flower | Fortress | Horns | Leaves | Orchids | Room | Trex |
|--- |---|---|---|---|---|---|---|---|
| PSNR (Test) | 25.50 | 27.82 | 31.08 | 27.47 | 21.46 | 20.51 | 30.33 | 26.26 |
| SSIM (Test) | 0.8282 | 0.8642 | 0.8809 | 0.8504 | 0.7589 | 0.6838 | 0.9388 | 0.8846 |
| LPIPS (Test) | 0.2270 | 0.1819 | 0.1853 | 0.2385 | 0.1992 | 0.2642 | 0.1898 | 0.2464 |
| PSNR (All) | 27.59 | 30.85 | 32.31 | 28.54 | 23.84 | 24.19 | 33.51 | 28.45 |
| SSIM (All) | 0.8755 | 0.9117 | 0.8963 | 0.8670 | 0.8303 | 0.7932 | 0.9555 | 0.9192 |
| LPIPS (All) | 0.1897 | 0.1375 | 0.1712 | 0.2214 | 0.1604 | 0.2099 | 0.1670 | 0.1996 |

### LLFF (NeRF-Real) (The same setting with the paper)

| | Fern | Flower | Fortress | Horns | Leaves | Orchids | Room | Trex |
|--- |---|---|---|---|---|---|---|---|
| PSNR (Test) | 25.55 | 27.85 | 31.19 | 27.56 | 21.45 | 20.46 | 30.54 | 26.41 |
| SSIM (Test) | 0.8325 | 0.8649 | 0.8842 | 0.8545 | 0.7607 | 0.6815 | 0.9409 | 0.8885 |
| LPIPS (Test) | 0.2195 | 0.1781 | 0.1802 | 0.2328 | 0.1952 | 0.2664 | 0.1860 | 0.2382 |
| PSNR (All) | 27.81 | 30.96 | 32.48 | 28.74 | 23.94 | 24.19 | 33.82 | 28.65 |
| SSIM (All) | 0.8811 | 0.9129 | 0.9000 | 0.8723 | 0.8341 | 0.7929 | 0.9574 | 0.9222 |
| LPIPS (All) | 0.1817 | 0.1328 | 0.1655 | 0.2144 | 0.1556 | 0.2111 | 0.1628 | 0.1900 |


### Tanks and Temples (Fair comparison with NeRF++)

| | M60 | Train | Truck | Playground | 
|--- |---|---|---|---|
| PSNR (Test) | 16.05 | 16.47 | 21.23 | 22.22 |
| SSIM (Test) | 0.6110 | 0.5271 | 0.6938 | 0.6668 | 
| LPIPS (Test) | 0.5196 | 0.5422 | 0.4456 | 0.4924 | 
| PSNR (All) | 24.11 | 21.08 | 23.34 | 24.58 |
| SSIM (All) | 0.7559 | 0.6285 | 0.7472 | 0.6932 |
| LPIPS (All) | 0.3817 | 0.4659 | 0.3743 | 0.4350 | 

### Tanks and Temples (The same setting with the paper)

| | M60 | Train | Truck | Playground | 
|--- |---|---|---|---|
| PSNR (Test) | 16.54 | 16.70 | 21.63 | 22.32 |
| SSIM (Test) | 0.6282 | 0.5487 | 0.7132 | 0.6771 | 
| LPIPS (Test) | 0.4971 | 0.5209 | 0.4071 | 0.4665 | 
| PSNR (All) | 24.82 | 21.70 | 23.73 | 24.63 |
| SSIM (All) | 0.7813 | 0.6588 | 0.7674 | 0.7091 |
| LPIPS (All) | 0.3488 | 0.4273 | 0.3359 | 0.4048 | 
