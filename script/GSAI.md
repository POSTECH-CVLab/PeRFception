# Document for GSAI setup

```bash
# Borrow the GPU for installing CUDA operations. 
srun -p 2080ti --gres=gpu:1 --pty /bin/bash -l

conda create -n perfception -c anaconda python=3.8 -y
conda activate perfception
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge -y
pip3 install imageio tqdm requests configargparse scikit-image imageio-ffmpeg piqa wandb pytorch_lightning==1.5.5 opencv-python gin-config gdown plyfile
pip3 install .

# Add your directory to write down the log. 
mkdir sbatch_log

# Register your wandb account
wandb login (~~)
```

```
# Instead of running the command you should get the agent code.
## First, create the sweep on online wandb repo.
## You can view the agent code on the sweep file.

wandb agent ~~
```


```
## Put your wandb agent here

#!/bin/sh

#SBATCH -J perfception # Job name
#SBATCH -o sbatch_log/pytorch-1gpu.%j.out # Name of stdout output file (%j expands to %jobId)
#SBATCH -p A100 # queue name or partiton name titanxp/titanrtx/2080ti
#SBATCH -t 3-00:00:00 # Run time (hh:mm:ss) - 1.5 hours
#SBATCH --gres=gpu:1 # number of gpus you want to use

#SBATCH --nodes=1
##SBATCH --exclude=n13
##SBTACH --nodelist=n12

##SBTACH --ntasks=1
##SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8

cd $SLURM_SUBMIT_DIR

echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_VERSION=$CUDA_VERSION"

srun -l /bin/hostname
srun -l /bin/pwd
srun -l /bin/date

module purge

echo "Start"
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export WANDB_SPAWN_METHOD=fork

wandb agent ~~

nvidia-smi
date
squeue --job $SLURM_JOBID

echo "##### END #####"


```