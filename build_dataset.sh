#!/bin/bash

OUT_DIR="$WORK_BASE/tpm/data/sudoku-extreme-1k-aug-1000"

python dataset/build_sudoku_dataset.py \
  --output-dir "$OUT_DIR" \
  --subsample-size 1000 \
  --num-aug 1000
