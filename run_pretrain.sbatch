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

RUN_NAME="sudoku_extreme_trm"
DATASET_NAME="sudoku-extreme-1k-aug-1000"

export WANDB_PROJECT="Tiny_progressive_models" #!FIX doesnt do anything rn

DATA_PATH="$WORK_BASE/tpm/data/$DATASET_NAME"
CHECKPOINT_PATH="$WORK_BASE/tpm/checkpoints/tpm/$RUN_NAME"

mkdir -p "$CHECKPOINT_PATH" logs

echo "RUN_NAME        = $RUN_NAME"
echo "DATA_PATH       = $DATA_PATH"
echo "CHECKPOINT_PATH = $CHECKPOINT_PATH"

python pretrain.py \
  arch=trm \
  data_paths="[$DATA_PATH]" \
  evaluators="[]" \
  epochs=50000 \
  eval_interval=10000 \
  checkpoint_interval=10000 \
  +checkpoint_path="$CHECKPOINT_PATH" \
  lr=1e-4 puzzle_emb_lr=1e-4 \
  weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=6 \
  +run_name="$RUN_NAME" \
  ema=True
