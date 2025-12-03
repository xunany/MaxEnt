#!/bin/sh
#SBATCH --job-name=dMRI
#SBATCH -N 1                                                ## requests on 1 node
#SBATCH -n 10                                               ## requests on 10 CPU
#SBATCH --gres=gpu:1                                        ## requests on 1 GPU
#SBATCH -p OOD_gpu_32gb                                     ## gpu, gpu-v100-16gb, gpu-v100-32gb, OOD_gpu_32gb
#SBATCH --exclude=node381                                   ## if use gpu-v100-32gb, don't use node 381. The cuda version is 10.1 on node 381
#SBATCH --nodelist=node365                                  ## if use OOD_gpu_32gb, only use node 365. This is the only real 32gb node
#SBATCH --output /work/xunan/logs/job%j.%N.out
#SBATCH --error /work/xunan/logs/job%j.%N.err

nvidia-smi                                                  ## this returns the cuda version information of the gpu

source /work/xunan/tools/miniconda3/etc/profile.d/conda.sh
conda activate torch_118

python /work/xunan/MaxEnt/HPC_Post_Process.py

