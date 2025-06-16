#!/bin/bash

# Configuration
MOSES_SCRIPTS=./moses_scripts # Set this to your Moses installation
OUTPUT_DIR="./dataset"        # Output directory for all processed files
src=hsb                       # Change to 'dsb' for Lower Sorbian, 'hsb' for Upper Sorbian
tgt=de
max_len=100 # Maximum sentence length (adjust as needed)

# BPE configuration
BPE_OPERATIONS=16000 # Number of BPE merge operations

# Directory structure
ORIGINAL_DIR="$OUTPUT_DIR/original"
MOSES_DIR="$OUTPUT_DIR/output_moses"
BPE_DIR="$OUTPUT_DIR/output_bpe"
DATA_BIN_DIR="$OUTPUT_DIR/fairseq"

# Split files (in original directory)
train_src="$ORIGINAL_DIR/train.${src}"
train_tgt="$ORIGINAL_DIR/train.${tgt}"
dev_src="$ORIGINAL_DIR/dev.${src}"
dev_tgt="$ORIGINAL_DIR/dev.${tgt}"
test_src="$ORIGINAL_DIR/test.${src}"
test_tgt="$ORIGINAL_DIR/test.${tgt}"

# Create output directories if they don't exist
mkdir -p "$ORIGINAL_DIR" "$MOSES_DIR" "$BPE_DIR" "$DATA_BIN_DIR"

# Check if split files exist
for split in train dev test; do
  for lang in $src $tgt; do
    file="$ORIGINAL_DIR/${split}.${lang}"
    if [ ! -f "$file" ]; then
      echo "Error: Split file $file not found!"
      exit 1
    fi
  done
done

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

# Step 1: Process each split separately through Moses pipeline
echo "Step 1-5: Processing train/dev/test splits through Moses pipeline..."

for split in train dev test; do
  echo "Processing $split split..."
  
  # Step 1: Normalize punctuation
  for lang in $src $tgt; do
    input_file="$ORIGINAL_DIR/${split}.${lang}"
    $MOSES_SCRIPTS/tokenizer/normalize-punctuation.perl -l $lang <"$input_file" >"$MOSES_DIR/${split}.norm.$lang"
  done
  
  # Step 2: Tokenize
  for lang in $src $tgt; do
    $MOSES_SCRIPTS/tokenizer/tokenizer.perl -a -l $lang <"$MOSES_DIR/${split}.norm.$lang" >"$MOSES_DIR/${split}.tok.$lang"
  done
  
  # Step 3: Clean corpus (only for train split to avoid removing dev/test data)
  if [ "$split" == "train" ]; then
    $MOSES_SCRIPTS/training/clean-corpus-n.perl "$MOSES_DIR/${split}.tok" $src $tgt "$MOSES_DIR/${split}.clean" 1 $max_len
  else
    # For dev/test, just copy tok files to clean files (no filtering)
    cp "$MOSES_DIR/${split}.tok.$src" "$MOSES_DIR/${split}.clean.$src"
    cp "$MOSES_DIR/${split}.tok.$tgt" "$MOSES_DIR/${split}.clean.$tgt"
  fi
done

# Step 4: Train truecaser on training data only
echo "Step 4: Training truecaser on training data..."
for lang in $src $tgt; do
  $MOSES_SCRIPTS/recaser/train-truecaser.perl -model "$MOSES_DIR/truecase-model.$lang" -corpus "$MOSES_DIR/train.tok.$lang"
done

# Step 5: Apply truecasing to all splits
echo "Step 5: Applying truecasing to all splits..."
for split in train dev test; do
  for lang in $src $tgt; do
    $MOSES_SCRIPTS/recaser/truecase.perl -model "$MOSES_DIR/truecase-model.$lang" <"$MOSES_DIR/${split}.clean.$lang" >"$MOSES_DIR/${split}.$lang"
  done
done

# Step 6: Learn BPE on training data only
echo "Step 6: Learning BPE on training data..."

# Check if training files are not empty
if [ ! -s "$MOSES_DIR/train.$src" ] || [ ! -s "$MOSES_DIR/train.$tgt" ]; then
  echo "Error: Training files are empty! Check the Moses preprocessing step."
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

# Step 7: Apply BPE to all splits
echo "Step 7: Applying BPE to all splits..."
for split in train dev test; do
  for lang in $src $tgt; do
    subword-nmt apply-bpe -c "$BPE_DIR/bpe.codes" <"$MOSES_DIR/$split.$lang" >"$BPE_DIR/$split.bpe.$lang"
  done
done

# Step 8: Create fairseq binary dataset
echo "Step 8: Creating fairseq binary dataset..."
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
echo "=== Original Split Statistics ==="
echo "Training sentences: $(wc -l <"$ORIGINAL_DIR/train.$src")"
echo "Development sentences: $(wc -l <"$ORIGINAL_DIR/dev.$src")"
echo "Test sentences: $(wc -l <"$ORIGINAL_DIR/test.$src")"
echo ""
echo "=== After Moses Processing ==="
echo "Training sentences: $(wc -l <"$MOSES_DIR/train.$src")"
echo "Development sentences: $(wc -l <"$MOSES_DIR/dev.$src")"
echo "Test sentences: $(wc -l <"$MOSES_DIR/test.$src")"
echo "BPE operations: $BPE_OPERATIONS"

# Optional: Clean up intermediate files (comment out if you want to keep them)
# rm -f "$MOSES_DIR"/*.norm.* "$MOSES_DIR"/*.tok.* "$MOSES_DIR"/*.clean.* "$BPE_DIR"/train_combined.txt

echo ""
echo "Preprocessing complete!"
echo ""
echo "=== Directory Structure ==="
echo "$ORIGINAL_DIR/ - Original train/dev/test split files"
echo "$MOSES_DIR/ - Moses preprocessing output, train/dev/test splits"
echo "$BPE_DIR/ - BPE codes and tokenized files"
echo "$DATA_BIN_DIR/ - Fairseq binary dataset (ready for training)"
echo ""
echo "=== Key Files for NMT Training ==="
echo "Fairseq binary data: $DATA_BIN_DIR/"
echo "BPE codes: $BPE_DIR/bpe.codes"
echo "Truecaser models: $MOSES_DIR/truecase-model.$src, $MOSES_DIR/truecase-model.$tgt"
echo ""
echo "Ready for NMT training!"
