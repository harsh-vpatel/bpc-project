#!/bin/bash

OUTPUT_FILE="results/out.txt"
HYPOTHESES_FILE="results/hypotheses.txt"
REFERENCES_FILE="dataset/output_moses_bpe/test.de"
SOURCE_FILE="dataset/output_moses_bpe/test.hsb"

fairseq-generate ./dataset/fairseq_bpe/ \
  --path ./checkpoints/sorbian_german_bpe/checkpoint_best.pt \
  --remove-bpe \
  --source-lang hsb \
  --target-lang de \
  --arch transformer_iwslt_de_en \
  --tokenizer moses | grep "D-" | sort -n -t'-' -k2 | cut -f 3 >"$HYPOTHESES_FILE"

sacrebleu "$REFERENCES_FILE" -i "$HYPOTHESES_FILE" -m bleu chrf --chrf-word-order 2
comet-score -s "$SOURCE_FILE" -t "$HYPOTHESES_FILE" -r "$REFERENCES_FILE"

