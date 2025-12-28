#!/bin/bash
#SBATCH -p h200ea
#SBATCH --account=h200ea
#SBATCH --gres=gpu:h200:2          
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=7:00:00
#SBATCH --job-name=trm
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source ~/.bashrc
micromamba activate tpm

export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1

bash run_pretrain.sh
