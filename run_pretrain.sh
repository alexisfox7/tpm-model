#!/bin/bash

# test
LAMBDA_PROG=0.1
RUN_NAME="sudoku_extreme_tpm_lambda${LAMBDA_PROG}"
DATASET_NAME="sudoku-extreme-1k-aug-1000"

# doesnt do anything rn
export WANDB_PROJECT="Tiny_progressive_models"

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
  arch.mlp_t=True arch.pos_encodings=none \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=6 \
  arch.loss.lambda_prog=$LAMBDA_PROG arch.loss.margin_m=0.0 arch.loss.phi_type=softplus \
  +run_name="$RUN_NAME" \
  ema=True