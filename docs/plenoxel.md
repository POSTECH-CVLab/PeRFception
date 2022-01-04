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