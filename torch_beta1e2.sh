#!/bin/sh
#SBATCH --job-name=dMRI
#SBATCH -N 1                    ## requests on 1 node
#SBATCH -n 24                   ## requests on 24 CPU
#SBATCH --gres=gpu:1            ## requests on 1 GPU
#SBATCH --output /work/xunan/logs/job%j.%N.out
#SBATCH --error /work/xunan/logs/job%j.%N.err
#SBATCH -p OOD_gpu_32gb         ## gpu-v100-32gb

nvidia-smi                      ## this returns the cuda version information of the gpu

source /work/xunan/tools/miniconda3/etc/profile.d/conda.sh
conda activate torch_118

python /work/xunan/MaxEnt/Real_data_torch_beta1e2.py

