#!/bin/bash

# Usage: ./prepare_morphology.sh [bpe|morfessor]
# Default: bpe

# Configuration
MOSES_SCRIPTS=./moses_scripts # Set this to your Moses installation
OUTPUT_DIR="./dataset"        # Output directory for all processed files
src=hsb                       # Change to 'dsb' for Lower Sorbian, 'hsb' for Upper Sorbian
tgt=de
max_len=100 # Maximum sentence length (adjust as needed)

# Segmentation method (bpe or morfessor)
SEGMENTATION_METHOD=${1:-bpe}

# BPE configuration
BPE_OPERATIONS=16000 # Number of BPE merge operations

# Morfessor configuration
MORFESSOR_ALGORITHM="recursive" # Options: recursive, viterbi, baseline
MORFESSOR_DAMPING=0.01          # Damping parameter for Morfessor
MORFESSOR_ALPHA=1.0             # Alpha parameter for Morfessor

# Validate segmentation method
if [[ "$SEGMENTATION_METHOD" != "bpe" && "$SEGMENTATION_METHOD" != "morfessor" ]]; then
  echo "Error: Invalid segmentation method '$SEGMENTATION_METHOD'. Use 'bpe' or 'morfessor'."
  exit 1
fi

echo "Using segmentation method: $SEGMENTATION_METHOD"

# Directory structure - different for each method
ORIGINAL_DIR="$OUTPUT_DIR/original"
MOSES_DIR="$OUTPUT_DIR/output_moses"
if [ "$SEGMENTATION_METHOD" == "bpe" ]; then
  SEGMENTATION_DIR="$OUTPUT_DIR/output_bpe"
  DATA_BIN_DIR="$OUTPUT_DIR/fairseq_bpe"
else
  SEGMENTATION_DIR="$OUTPUT_DIR/output_morfessor"
  DATA_BIN_DIR="$OUTPUT_DIR/fairseq_morfessor"
fi

# Split files (in original directory)
train_src="$ORIGINAL_DIR/train.${src}"
train_tgt="$ORIGINAL_DIR/train.${tgt}"
dev_src="$ORIGINAL_DIR/dev.${src}"
dev_tgt="$ORIGINAL_DIR/dev.${tgt}"
test_src="$ORIGINAL_DIR/test.${src}"
test_tgt="$ORIGINAL_DIR/test.${tgt}"

# Create output directories if they don't exist
mkdir -p "$ORIGINAL_DIR" "$MOSES_DIR" "$SEGMENTATION_DIR" "$DATA_BIN_DIR"

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

# Check dependencies based on segmentation method
if [ "$SEGMENTATION_METHOD" == "bpe" ]; then
  if ! command -v subword-nmt &>/dev/null; then
    echo "Error: subword-nmt not found! Install with: pip install subword-nmt"
    exit 1
  fi
else
  if ! python -c "import morfessor" &>/dev/null; then
    echo "Error: morfessor not found! Install with: pip install morfessor"
    exit 1
  fi
fi

# Check if fairseq is installed
if ! command -v fairseq-preprocess &>/dev/null; then
  echo "Error: fairseq not found! Install with: pip install fairseq"
  exit 1
fi

# Clean up any previous runs in output directories
rm -f "$MOSES_DIR"/*
if [ "$SEGMENTATION_METHOD" == "bpe" ]; then
  rm -f "$SEGMENTATION_DIR"/*
else
  # For morfessor, preserve existing models but clean other files
  find "$SEGMENTATION_DIR" -type f ! -name "morfessor_model.*.bin" -delete
fi

echo "Starting preprocessing for ${src}-${tgt} with ${SEGMENTATION_METHOD}..."

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

# Step 6: Learn segmentation on training data only
if [ "$SEGMENTATION_METHOD" == "bpe" ]; then
  echo "Step 6: Learning BPE on training data..."

  # Check if training files are not empty
  if [ ! -s "$MOSES_DIR/train.$src" ] || [ ! -s "$MOSES_DIR/train.$tgt" ]; then
    echo "Error: Training files are empty! Check the Moses preprocessing step."
    exit 1
  fi

  # Combine training data and learn BPE
  cat "$MOSES_DIR/train.$src" "$MOSES_DIR/train.$tgt" >"$SEGMENTATION_DIR/train_combined.txt"
  echo "Combined training data has $(wc -l <"$SEGMENTATION_DIR/train_combined.txt") lines"

  # Learn BPE with minimum frequency threshold
  subword-nmt learn-bpe -s $BPE_OPERATIONS --min-frequency 2 <"$SEGMENTATION_DIR/train_combined.txt" >"$SEGMENTATION_DIR/bpe.codes"

  # Check if BPE codes were created successfully
  if [ ! -s "$SEGMENTATION_DIR/bpe.codes" ]; then
    echo "Error: BPE codes file is empty or was not created!"
    echo "This might happen with very small datasets. Try reducing BPE_OPERATIONS."
    exit 1
  fi

  echo "BPE codes learned successfully ($(wc -l <"$SEGMENTATION_DIR/bpe.codes") operations)"

else
  echo "Step 6: Learning Morfessor segmentation on training data..."

  # Check if training files are not empty
  if [ ! -s "$MOSES_DIR/train.$src" ] || [ ! -s "$MOSES_DIR/train.$tgt" ]; then
    echo "Error: Training files are empty! Check the Moses preprocessing step."
    exit 1
  fi

  # Learn Morfessor models for each language separately
  for lang in $src $tgt; do
    if [ -f "$SEGMENTATION_DIR/morfessor_model.$lang.bin" ]; then
      echo "Morfessor model for $lang already exists, skipping training..."
    else
      echo "Training Morfessor model for $lang..."
      morfessor-train \
        -s "$SEGMENTATION_DIR/morfessor_model.$lang.bin" \
        -d ones \
        "$MOSES_DIR/train.$lang"
    fi
  done

  # Check if Morfessor models were created successfully
  for lang in $src $tgt; do
    if [ ! -f "$SEGMENTATION_DIR/morfessor_model.$lang.bin" ]; then
      echo "Error: Morfessor model for $lang was not created!"
      exit 1
    fi
  done

  echo "Morfessor models trained successfully for both languages"
fi

# Step 7: Apply segmentation to all splits
echo "Step 7: Applying ${SEGMENTATION_METHOD} to all splits..."

if [ "$SEGMENTATION_METHOD" == "bpe" ]; then
  for split in train dev test; do
    for lang in $src $tgt; do
      subword-nmt apply-bpe -c "$SEGMENTATION_DIR/bpe.codes" <"$MOSES_DIR/$split.$lang" >"$SEGMENTATION_DIR/$split.bpe.$lang"
    done
  done

  # Set file extension for fairseq preprocessing
  SEGMENTATION_EXT="bpe"
else
  for split in train dev test; do
    for lang in $src $tgt; do
      echo "Applying Morfessor segmentation to $split.$lang..."
      morfessor-segment \
        -l "$SEGMENTATION_DIR/morfessor_model.$lang.bin" \
        --output-format-separator "@@ " \
        --output-newlines \
        --output-format "{analysis} " \
        "$MOSES_DIR/$split.$lang" \
        >"$SEGMENTATION_DIR/$split.morfessor.$lang"
    done
  done

  # Set file extension for fairseq preprocessing
  SEGMENTATION_EXT="morfessor"
fi

# Step 8: Create fairseq binary dataset
echo "Step 8: Creating fairseq binary dataset..."
fairseq-preprocess \
  --source-lang $src --target-lang $tgt \
  --trainpref "$SEGMENTATION_DIR/train.$SEGMENTATION_EXT" \
  --validpref "$SEGMENTATION_DIR/dev.$SEGMENTATION_EXT" \
  --testpref "$SEGMENTATION_DIR/test.$SEGMENTATION_EXT" \
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

if [ "$SEGMENTATION_METHOD" == "bpe" ]; then
  echo "BPE operations: $BPE_OPERATIONS"
else
  echo "Morfessor algorithm: $MORFESSOR_ALGORITHM"
  echo "Morfessor damping: $MORFESSOR_DAMPING"
  echo "Morfessor alpha: $MORFESSOR_ALPHA"
fi

# Optional: Clean up intermediate files (comment out if you want to keep them)
# rm -f "$MOSES_DIR"/*.norm.* "$MOSES_DIR"/*.tok.* "$MOSES_DIR"/*.clean.* "$SEGMENTATION_DIR"/train_combined.txt

echo ""
echo "Preprocessing complete!"
echo ""
echo "=== Directory Structure ==="
echo "$ORIGINAL_DIR/ - Original train/dev/test split files"
echo "$MOSES_DIR/ - Moses preprocessing output, train/dev/test splits"
echo "$SEGMENTATION_DIR/ - ${SEGMENTATION_METHOD^^} codes/models and tokenized files"
echo "$DATA_BIN_DIR/ - Fairseq binary dataset (ready for training)"
echo ""
echo "=== Key Files for NMT Training ==="
echo "Fairseq binary data: $DATA_BIN_DIR/"
if [ "$SEGMENTATION_METHOD" == "bpe" ]; then
  echo "BPE codes: $SEGMENTATION_DIR/bpe.codes"
else
  echo "Morfessor models: $SEGMENTATION_DIR/morfessor_model.$src.bin, $SEGMENTATION_DIR/morfessor_model.$tgt.bin"
fi
echo "Truecaser models: $MOSES_DIR/truecase-model.$src, $MOSES_DIR/truecase-model.$tgt"
echo ""
echo "Ready for NMT training with $SEGMENTATION_METHOD segmentation!"
