#!/bin/bash

# Configuration
MOSES_SCRIPTS=./moses_scripts # Set this to your Moses installation
OUTPUT_DIR="./dataset"        # Output directory for all processed files
src=hsb                       # Change to 'dsb' for Lower Sorbian, 'hsb' for Upper Sorbian
tgt=de
max_len=100 # Maximum sentence length (adjust as needed)

# Split ratios
TRAIN_RATIO=0.8
DEV_RATIO=0.1
TEST_RATIO=0.1

# BPE configuration
BPE_OPERATIONS=16000 # Number of BPE merge operations

# Directory structure
ORIGINAL_DIR="$OUTPUT_DIR/original"
MOSES_DIR="$OUTPUT_DIR/output_moses"
BPE_DIR="$OUTPUT_DIR/output_bpe"
DATA_BIN_DIR="$OUTPUT_DIR/fairseq"

# Input files (in original directory)
input_src="$ORIGINAL_DIR/input.${src}"
input_tgt="$ORIGINAL_DIR/input.${tgt}"

# Create output directories if they don't exist
mkdir -p "$ORIGINAL_DIR" "$MOSES_DIR" "$BPE_DIR" "$DATA_BIN_DIR"

# Check if input files exist
if [ ! -f "$input_src" ] || [ ! -f "$input_tgt" ]; then
  echo "Error: Input files $input_src or $input_tgt not found!"
  exit 1
fi

# Check if subword-nmt is installed
if ! command -v subword-nmt &>/dev/null; then
  echo "Error: subword-nmt not found! Install with: pip install subword-nmt"
  exit 1
fi

# Check if fairseq is installed
if ! command -v fairseq-preprocess &>/dev/null; then
  echo "Error: fairseq not found! Install with: pip install fairseq"
  exit 1
fi

# Check if bc is installed (needed for floating point arithmetic)
if ! command -v bc &>/dev/null; then
  echo "Error: bc not found! Install with: apt-get install bc (Ubuntu) or brew install bc (macOS)"
  exit 1
fi

# Clean up any previous runs in output directories
rm -f "$MOSES_DIR"/* "$BPE_DIR"/*

echo "Starting preprocessing for ${src}-${tgt}..."

# Step 1: Normalize punctuation
echo "Step 1: Normalizing punctuation..."
for lang in $src $tgt; do
  if [ $lang == $src ]; then
    input_file=$input_src
  else
    input_file=$input_tgt
  fi

  $MOSES_SCRIPTS/tokenizer/normalize-punctuation.perl -l $lang <$input_file >"$MOSES_DIR/corpus.norm.$lang"
done

# Step 2: Tokenize
echo "Step 2: Tokenizing..."
for lang in $src $tgt; do
  $MOSES_SCRIPTS/tokenizer/tokenizer.perl -a -l $lang <"$MOSES_DIR/corpus.norm.$lang" >"$MOSES_DIR/corpus.tok.$lang"
done

# Step 3: Clean corpus (remove sentences that are too long/short or misaligned)
echo "Step 3: Cleaning corpus..."
$MOSES_SCRIPTS/training/clean-corpus-n.perl "$MOSES_DIR/corpus.tok" $src $tgt "$MOSES_DIR/corpus.clean" 1 $max_len

# Step 4: Train truecaser
echo "Step 4: Training truecaser..."
for lang in $src $tgt; do
  $MOSES_SCRIPTS/recaser/train-truecaser.perl -model "$MOSES_DIR/truecase-model.$lang" -corpus "$MOSES_DIR/corpus.tok.$lang"
done

# Step 5: Apply truecasing
echo "Step 5: Applying truecasing..."
for lang in $src $tgt; do
  $MOSES_SCRIPTS/recaser/truecase.perl -model "$MOSES_DIR/truecase-model.$lang" <"$MOSES_DIR/corpus.clean.$lang" >"$MOSES_DIR/corpus.tc.$lang"
done

# Create final Moses output
echo "Creating final Moses parallel corpus..."
paste "$MOSES_DIR/corpus.tc.$src" "$MOSES_DIR/corpus.tc.$tgt" >"$MOSES_DIR/final_corpus.txt"

# Step 6: Split into train/dev/test
echo "Step 6: Splitting into train/dev/test sets..."
total_lines=$(wc -l <"$MOSES_DIR/corpus.tc.$src")
train_lines=$(echo "$total_lines * 0.8" | bc | cut -d. -f1)
dev_lines=$(echo "$total_lines * 0.1" | bc | cut -d. -f1)

echo "Total lines: $total_lines, Train: $train_lines, Dev: $dev_lines"

# Split source language
head -n "$train_lines" "$MOSES_DIR/corpus.tc.$src" >"$MOSES_DIR/train.$src"
tail -n +$((train_lines + 1)) "$MOSES_DIR/corpus.tc.$src" | head -n "$dev_lines" >"$MOSES_DIR/dev.$src"
tail -n +$((train_lines + dev_lines + 1)) "$MOSES_DIR/corpus.tc.$src" >"$MOSES_DIR/test.$src"

# Split target language
head -n "$train_lines" "$MOSES_DIR/corpus.tc.$tgt" >"$MOSES_DIR/train.$tgt"
tail -n +$((train_lines + 1)) "$MOSES_DIR/corpus.tc.$tgt" | head -n "$dev_lines" >"$MOSES_DIR/dev.$tgt"
tail -n +$((train_lines + dev_lines + 1)) "$MOSES_DIR/corpus.tc.$tgt" >"$MOSES_DIR/test.$tgt"

# Step 7: Learn BPE on training data only
echo "Step 7: Learning BPE on training data..."

# Check if training files are not empty
if [ ! -s "$MOSES_DIR/train.$src" ] || [ ! -s "$MOSES_DIR/train.$tgt" ]; then
  echo "Error: Training files are empty! Check the splitting step."
  exit 1
fi

# Combine training data and learn BPE
cat "$MOSES_DIR/train.$src" "$MOSES_DIR/train.$tgt" >"$BPE_DIR/train_combined.txt"
echo "Combined training data has $(wc -l <"$BPE_DIR/train_combined.txt") lines"

# Learn BPE with minimum frequency threshold
subword-nmt learn-bpe -s $BPE_OPERATIONS --min-frequency 2 <"$BPE_DIR/train_combined.txt" >"$BPE_DIR/bpe.codes"

# Check if BPE codes were created successfully
if [ ! -s "$BPE_DIR/bpe.codes" ]; then
  echo "Error: BPE codes file is empty or was not created!"
  echo "This might happen with very small datasets. Try reducing BPE_OPERATIONS."
  exit 1
fi

echo "BPE codes learned successfully ($(wc -l <"$BPE_DIR/bpe.codes") operations)"

# Step 8: Apply BPE to all splits
echo "Step 8: Applying BPE to all splits..."
for split in train dev test; do
  for lang in $src $tgt; do
    subword-nmt apply-bpe -c "$BPE_DIR/bpe.codes" <"$MOSES_DIR/$split.$lang" >"$BPE_DIR/$split.bpe.$lang"
  done
done

# Step 9: Create fairseq binary dataset
echo "Step 9: Creating fairseq binary dataset..."
fairseq-preprocess \
  --source-lang $src --target-lang $tgt \
  --trainpref "$BPE_DIR/train.bpe" \
  --validpref "$BPE_DIR/dev.bpe" \
  --testpref "$BPE_DIR/test.bpe" \
  --destdir "$DATA_BIN_DIR" \
  --workers 4

echo "Fairseq binary dataset created successfully!"

# Show statistics
echo "=== Preprocessing Statistics ==="
echo "Original sentences: $(wc -l <$input_src)"
echo "After cleaning: $(wc -l <"$MOSES_DIR/corpus.clean.$src")"
echo "Final sentences: $(wc -l <"$MOSES_DIR/corpus.tc.$src")"
echo ""
echo "=== Split Statistics ==="
echo "Training sentences: $(wc -l <"$MOSES_DIR/train.$src")"
echo "Development sentences: $(wc -l <"$MOSES_DIR/dev.$src")"
echo "Test sentences: $(wc -l <"$MOSES_DIR/test.$src")"
echo "BPE operations: $BPE_OPERATIONS"

# Optional: Clean up intermediate files (comment out if you want to keep them)
# rm -f "$MOSES_DIR"/corpus.norm.* "$MOSES_DIR"/corpus.tok.* "$MOSES_DIR"/corpus.clean.* "$MOSES_DIR"/corpus.tc.* "$BPE_DIR"/train_combined.txt

echo ""
echo "Preprocessing complete!"
echo ""
echo "=== Directory Structure ==="
echo "$ORIGINAL_DIR/ - Original input files"
echo "$MOSES_DIR/ - Moses preprocessing output, train/dev/test splits"
echo "$BPE_DIR/ - BPE codes and tokenized files"
echo "$DATA_BIN_DIR/ - Fairseq binary dataset (ready for training)"
echo ""
echo "=== Key Files for NMT Training ==="
echo "Fairseq binary data: $DATA_BIN_DIR/"
echo "BPE codes: $BPE_DIR/bpe.codes"
echo "Truecaser models: $MOSES_DIR/truecase-model.$src, $MOSES_DIR/truecase-model.$tgt"
echo ""
echo "=== Training Command ==="
echo "CUDA_VISIBLE_DEVICES=0,1 fairseq-train \\"
echo "    $DATA_BIN_DIR \\"
echo "    --arch transformer_iwslt_de_en \\"
echo "    --share-decoder-input-output-embed \\"
echo "    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \\"
echo "    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \\"
echo "    --dropout 0.3 --weight-decay 0.0001 \\"
echo "    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \\"
echo "    --max-tokens 8192 \\"
echo "    --eval-bleu \\"
echo "    --eval-bleu-args '{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}' \\"
echo "    --eval-bleu-detok moses \\"
echo "    --eval-bleu-remove-bpe \\"
echo "    --eval-bleu-print-samples \\"
echo "    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \\"
echo "    --save-dir checkpoints/sorbian_german_transformer \\"
echo "    --ddp-backend=pytorch_ddp \\"
echo "    --distributed-world-size 2 \\"
echo "    --distributed-port 12345"
echo ""
echo "Ready for NMT training!"
