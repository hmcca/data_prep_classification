#!/bin/bash
#SBATCH -A MED116_MDE
#SBATCH -J tokenize
#SBATCH -t  2:00:00
#SBATCH -q debug
#SBATCH -p batch-spi 
#SBATCH -N 1
#SBATCH -C nvme
#SBATCH --output=/gpfs/arx2/med116_mde/proj-shared/RITM0276466/new_data/logs/tokenize_%j.out
#SBATCH --error=/gpfs/arx2/med116_mde/proj-shared/RITM0276466/new_data/logs/tokenize_%j.err

source /sw/summit/mde/med116_mde/gounley1/env600vllm.sh

# export HF_HOME=/lustre/orion/proj-shared/med117/gounley1/hfhome
# export HF_LOCAL_HOME=/lustre/orion/proj-shared/med117/gounley1/hfhomelocal
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1

export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=3442
export NCCL_SOCKET_IFNAME=hsn0

srun -N1 -n8 -c7 --gpus-per-task=1 --gpu-bind=closest python /sw/summit/mde/med116_mde/shivannaa/data_prep_scripts/tokenize_text_col.py
