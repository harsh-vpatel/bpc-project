#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 fairseq-train \
    $DATA_BIN \
    --arch transformer \
    --encoder-embed-dim 512 --decoder-embed-dim 512 \
    --encoder-ffn-embed-dim 2048 --decoder-ffn-embed-dim 2048 \
    --encoder-layers 6 --decoder-layers 6 \
    --encoder-attention-heads 8 --decoder-attention-heads 8 \
    --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 1e-4 --lr-scheduler inverse_sqrt --warmup-updates 8000 \
    --dropout 0.1 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 12288 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir checkpoints/sorbian_german_transformer_base \
    --ddp-backend=pytorch_ddp \
    --distributed-world-size 2 \
    --distributed-port 12345 \
    --update-freq 1
