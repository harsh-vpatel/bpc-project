#!/bin/bash

# Unified training script for BPC neural machine translation project
# Usage: ./train.sh [dataset_dir] [checkpoint_dir] [gpu_id]
# Examples:
#   ./train.sh ./dataset/fairseq_bpe_hsb-de/ checkpoints/sorbian_german_bpe 0
#   ./train.sh ./dataset/fairseq_morfessor_bpe_de-hsb/ checkpoints/german_sorbian_morfessor_bpe 1

DATASET_DIR=${1:-"./dataset/fairseq_bpe/"}
CHECKPOINT_DIR=${2:-"checkpoints/sorbian_german_bpe"}
GPU_ID=${3:-"0"}

# Check if dataset directory exists
if [[ ! -d "$DATASET_DIR" ]]; then
    echo "Error: Dataset directory '$DATASET_DIR' does not exist."
    exit 1
fi

# Print configuration
echo "Training configuration:"
echo "  Dataset directory: $DATASET_DIR"
echo "  Checkpoint directory: $CHECKPOINT_DIR"
echo "  GPU ID: $GPU_ID"
echo

# Run training
CUDA_VISIBLE_DEVICES=$GPU_ID fairseq-train \
  $DATASET_DIR \
  --arch transformer_iwslt_de_en \
  --wandb-project "BPC Project" \
  --max-epoch 50 \
  --patience 10 \
  --save-interval 1 \
  --validate-interval 1 \
  --share-decoder-input-output-embed \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --dropout 0.3 --weight-decay 0.0001 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --max-tokens 4096 \
  --eval-bleu \
  --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
  --eval-bleu-detok moses \
  --eval-bleu-remove-bpe \
  --eval-bleu-print-samples \
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
  --save-dir $CHECKPOINT_DIR