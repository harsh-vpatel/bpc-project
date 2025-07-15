#!/bin/bash

# Usage: ./prepare.sh [direction] [morfessor] [--mono] [--rnn-tagger]
# Direction: hsb-de (default) or de-hsb
# Morfessor: Optional, apply Morfessor segmentation before BPE
# --mono: Optional, use monolingual data for BPE/Morfessor training
# --rnn-tagger: Optional, apply RNN-Tagger before Morfessor/BPE

# Parse arguments
DIRECTION=${1:-hsb-de}
USE_MORFESSOR_ARG=${2:-}
USE_MONO_ARG=${3:-}
USE_RNN_TAGGER_ARG=${4:-}

# Validate direction argument
if [[ "$DIRECTION" != "hsb-de" && "$DIRECTION" != "de-hsb" ]]; then
  echo "Error: Invalid direction '$DIRECTION'. Use 'hsb-de' or 'de-hsb'."
  exit 1
fi

# Set source and target languages based on direction
if [[ "$DIRECTION" == "hsb-de" ]]; then
  src=hsb
  tgt=de
else
  src=de
  tgt=hsb
fi

# Configuration
MOSES_SCRIPTS=./moses_scripts # Set this to your Moses installation
OUTPUT_DIR="./dataset"        # Output directory for all processed files
max_len=100                   # Maximum sentence length (adjust as needed)

# Use Morfessor preprocessing (optional)
USE_MORFESSOR=${USE_MORFESSOR_ARG:-}
if [[ "$USE_MORFESSOR" == "morfessor" ]]; then
  USE_MORFESSOR=true
else
  USE_MORFESSOR=false
fi

# BPE configuration
BPE_OPERATIONS=16000 # Number of BPE merge operations

# Morfessor configuration
MORFESSOR_ALGORITHM="recursive" # Options: recursive, viterbi, baseline
MORFESSOR_DAMPING=0.01          # Damping parameter for Morfessor
MORFESSOR_ALPHA=1.0             # Alpha parameter for Morfessor

# Validate morfessor argument
if [[ -n "$USE_MORFESSOR_ARG" && "$USE_MORFESSOR_ARG" != "morfessor" && "$USE_MORFESSOR_ARG" != "--mono" && "$USE_MORFESSOR_ARG" != "--rnn-tagger" ]]; then
  echo "Error: Invalid morfessor argument '$USE_MORFESSOR_ARG'. Use 'morfessor' or leave empty for BPE-only."
  exit 1
fi

# Handle --mono and --rnn-tagger flags in different argument positions
USE_MONO=false
USE_RNN_TAGGER=false

# Check all arguments for flags
for arg in "$USE_MORFESSOR_ARG" "$USE_MONO_ARG" "$USE_RNN_TAGGER_ARG"; do
  if [[ "$arg" == "--mono" ]]; then
    USE_MONO=true
  elif [[ "$arg" == "--rnn-tagger" ]]; then
    USE_RNN_TAGGER=true
  fi
done

# Clean up morfessor argument
if [[ "$USE_MORFESSOR_ARG" == "--mono" || "$USE_MORFESSOR_ARG" == "--rnn-tagger" ]]; then
  USE_MORFESSOR_ARG=""
fi

# Validate mono and rnn-tagger arguments
if [[ -n "$USE_MONO_ARG" && "$USE_MONO_ARG" != "--mono" && "$USE_MONO_ARG" != "--rnn-tagger" ]]; then
  echo "Error: Invalid mono argument '$USE_MONO_ARG'. Use '--mono' or leave empty."
  exit 1
fi

if [[ -n "$USE_RNN_TAGGER_ARG" && "$USE_RNN_TAGGER_ARG" != "--rnn-tagger" ]]; then
  echo "Error: Invalid rnn-tagger argument '$USE_RNN_TAGGER_ARG'. Use '--rnn-tagger' or leave empty."
  exit 1
fi

echo "Processing direction: $DIRECTION ($src -> $tgt)"
if [[ "$USE_RNN_TAGGER" == "true" ]]; then
  echo "Using RNN-Tagger preprocessing"
fi
if [[ "$USE_MORFESSOR" == "true" ]]; then
  echo "Using Morfessor + BPE segmentation pipeline"
else
  echo "Using BPE-only segmentation pipeline"
fi
if [[ "$USE_MONO" == "true" ]]; then
  echo "Using monolingual data for BPE/Morfessor training"
else
  echo "Using training data for BPE/Morfessor training"
fi

# Directory structure - add direction to final binary dataset
ORIGINAL_DIR="$OUTPUT_DIR/original"
MOSES_DIR="$OUTPUT_DIR/output_moses"
if [[ "$USE_RNN_TAGGER" == "true" ]]; then
  RNN_DIR="$OUTPUT_DIR/output_rnn_tagger"
fi
if [[ "$USE_MORFESSOR" == "true" ]]; then
  MORFESSOR_DIR="$OUTPUT_DIR/output_morfessor"
  if [[ "$USE_RNN_TAGGER" == "true" ]]; then
    BPE_DIR="$OUTPUT_DIR/output_rnn_morfessor_bpe"
    DATA_BIN_DIR="$OUTPUT_DIR/fairseq_rnn_morfessor_bpe_${DIRECTION}"
  else
    BPE_DIR="$OUTPUT_DIR/output_morfessor_bpe"
    DATA_BIN_DIR="$OUTPUT_DIR/fairseq_morfessor_bpe_${DIRECTION}"
  fi
else
  if [[ "$USE_RNN_TAGGER" == "true" ]]; then
    BPE_DIR="$OUTPUT_DIR/output_rnn_bpe"
    DATA_BIN_DIR="$OUTPUT_DIR/fairseq_rnn_bpe_${DIRECTION}"
  else
    BPE_DIR="$OUTPUT_DIR/output_bpe"
    DATA_BIN_DIR="$OUTPUT_DIR/fairseq_bpe_${DIRECTION}"
  fi
fi

# Split files (in original directory)
train_src="$ORIGINAL_DIR/train.${src}"
train_tgt="$ORIGINAL_DIR/train.${tgt}"
dev_src="$ORIGINAL_DIR/dev.${src}"
dev_tgt="$ORIGINAL_DIR/dev.${tgt}"
test_src="$ORIGINAL_DIR/test.${src}"
test_tgt="$ORIGINAL_DIR/test.${tgt}"

# Monolingual data file (always Sorbian)
mono_src_gz="$ORIGINAL_DIR/mono.hsb.gz"
mono_src="$ORIGINAL_DIR/mono.hsb"

# Create output directories if they don't exist
if [[ "$USE_RNN_TAGGER" == "true" ]]; then
  mkdir -p "$ORIGINAL_DIR" "$MOSES_DIR" "$RNN_DIR" "$BPE_DIR" "$DATA_BIN_DIR"
fi
if [[ "$USE_MORFESSOR" == "true" ]]; then
  mkdir -p "$ORIGINAL_DIR" "$MOSES_DIR" "$MORFESSOR_DIR" "$BPE_DIR" "$DATA_BIN_DIR"
else
  mkdir -p "$ORIGINAL_DIR" "$MOSES_DIR" "$BPE_DIR" "$DATA_BIN_DIR"
fi

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

# Check if monolingual file exists when --mono flag is used
if [[ "$USE_MONO" == "true" ]]; then
  if [ -f "$mono_src_gz" ]; then
    echo "Found gzipped monolingual file, decompressing..."
    gunzip -c "$mono_src_gz" >"$mono_src"
    if [ $? -ne 0 ]; then
      echo "Error: Failed to decompress $mono_src_gz"
      exit 1
    fi
  elif [ ! -f "$mono_src" ]; then
    echo "Error: Monolingual file $mono_src or $mono_src_gz not found!"
    exit 1
  fi
  if [ ! -s "$mono_src" ]; then
    echo "Error: Monolingual file $mono_src is empty!"
    exit 1
  fi
fi

# Check dependencies
if ! command -v subword-nmt &>/dev/null; then
  echo "Error: subword-nmt not found! Install with: pip install subword-nmt"
  exit 1
fi

if [[ "$USE_MORFESSOR" == "true" ]] && ! python -c "import morfessor" &>/dev/null; then
  echo "Error: morfessor not found! Install with: pip install morfessor"
  exit 1
fi

if [[ "$USE_RNN_TAGGER" == "true" ]]; then
  if [ ! -f "./RNNTagger/cmd/rnn-tagger-german.sh" ] || [ ! -f "./RNNTagger/cmd/rnn-tagger-upper-sorbian.sh" ]; then
    echo "Error: RNN-Tagger scripts not found! Check ./RNNTagger/cmd/ directory."
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
rm -f "$BPE_DIR"/*
if [[ "$USE_RNN_TAGGER" == "true" ]]; then
  rm -f "$RNN_DIR"/*
fi
if [[ "$USE_MORFESSOR" == "true" ]]; then
  # For morfessor, preserve existing models but clean other files
  find "$MORFESSOR_DIR" -type f ! -name "morfessor_model.*.bin" -delete 2>/dev/null || true
fi

if [[ "$USE_RNN_TAGGER" == "true" && "$USE_MORFESSOR" == "true" ]]; then
  echo "Starting preprocessing for ${src}-${tgt} with RNN-Tagger + Morfessor + BPE..."
elif [[ "$USE_RNN_TAGGER" == "true" ]]; then
  echo "Starting preprocessing for ${src}-${tgt} with RNN-Tagger + BPE..."
elif [[ "$USE_MORFESSOR" == "true" ]]; then
  echo "Starting preprocessing for ${src}-${tgt} with Morfessor + BPE..."
else
  echo "Starting preprocessing for ${src}-${tgt} with BPE..."
fi

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

# Step 4: Apply RNN-Tagger (if enabled)
if [[ "$USE_RNN_TAGGER" == "true" ]]; then
  echo "Step 4: Applying RNN-Tagger to all splits..."

  # Apply RNN-Tagger to all splits
  for split in train dev test; do
    for lang in $src $tgt; do
      echo "Applying RNN-Tagger to $split.$lang..."

      # Choose appropriate RNN-Tagger script
      if [[ "$lang" == "hsb" ]]; then
        RNN_SCRIPT="cmd/rnn-tagger-upper-sorbian.sh"
      else
        RNN_SCRIPT="cmd/rnn-tagger-german.sh"
      fi

      # Apply RNN-Tagger
      cd RNNTagger
      bash "${RNN_SCRIPT}" "../$MOSES_DIR/$split.clean.$lang" >"../$RNN_DIR/$split.rnn.$lang"
      cd ..

      sed -i -e ':a' -e 'N' -e '$!ba' -e 's/\n\n/\x00/g' -e 's/\n//g' -e 's/\x00/\n/g' "../$RNN_DIR/$split.rnn.$lang"
      # Check if RNN-Tagger output is valid
      if [ ! -s "$RNN_DIR/$split.rnn.$lang" ]; then
        echo "Error: RNN-Tagger output for $split.$lang is empty!"
        exit 1
      fi
    done
  done

  # Process monolingual data through RNN-Tagger if --mono is also used
  if [[ "$USE_MONO" == "true" ]]; then
    echo "Processing monolingual data through RNN-Tagger..."
    cd RNNTagger
    bash "cmd/rnn-tagger-upper-sorbian.sh" "../$mono_src" >"../$RNN_DIR/mono.rnn.hsb"
    cd ..

    # Check if monolingual RNN-Tagger output is valid
    if [ ! -s "$RNN_DIR/mono.rnn.hsb" ]; then
      echo "Error: RNN-Tagger output for monolingual data is empty!"
      exit 1
    fi

    # Update mono_src to point to RNN-tagged version
    mono_src_rnn="$RNN_DIR/mono.rnn.hsb"
  fi

  echo "RNN-Tagger processing completed successfully"
fi

# Step 5: Train truecaser on training data
echo "Step 5: Training truecaser on training data..."
for lang in $src $tgt; do
  if [[ "$USE_RNN_TAGGER" == "true" ]]; then
    # Train truecaser on RNN-tagged data
    $MOSES_SCRIPTS/recaser/train-truecaser.perl -model "$MOSES_DIR/truecase-model.$lang" -corpus "$RNN_DIR/train.rnn.$lang"
  else
    # Train truecaser on tokenized data
    $MOSES_SCRIPTS/recaser/train-truecaser.perl -model "$MOSES_DIR/truecase-model.$lang" -corpus "$MOSES_DIR/train.tok.$lang"
  fi
done

# Step 6: Apply truecasing to all splits
echo "Step 6: Applying truecasing to all splits..."
for split in train dev test; do
  for lang in $src $tgt; do
    if [[ "$USE_RNN_TAGGER" == "true" ]]; then
      # Apply truecasing to RNN-tagged data
      $MOSES_SCRIPTS/recaser/truecase.perl -model "$MOSES_DIR/truecase-model.$lang" <"$RNN_DIR/${split}.rnn.$lang" >"$RNN_DIR/${split}.rnn.true.$lang"
    else
      # Apply truecasing to clean data
      $MOSES_SCRIPTS/recaser/truecase.perl -model "$MOSES_DIR/truecase-model.$lang" <"$MOSES_DIR/${split}.clean.$lang" >"$MOSES_DIR/${split}.$lang"
    fi
  done
done

# Apply truecasing to monolingual data if both --mono and --rnn-tagger are used
if [[ "$USE_MONO" == "true" && "$USE_RNN_TAGGER" == "true" ]]; then
  echo "Applying truecasing to monolingual data..."
  $MOSES_SCRIPTS/recaser/truecase.perl -model "$MOSES_DIR/truecase-model.hsb" <"$RNN_DIR/mono.rnn.hsb" >"$RNN_DIR/mono.rnn.true.hsb"
  # Update mono_src to point to truecased version
  mono_src_rnn="$RNN_DIR/mono.rnn.true.hsb"
fi

# Update file extensions and input directories
if [[ "$USE_RNN_TAGGER" == "true" ]]; then
  TRUECASED_INPUT_DIR="$RNN_DIR"
  TRUECASED_EXT="rnn.true"
else
  TRUECASED_INPUT_DIR="$MOSES_DIR"
  TRUECASED_EXT=""
fi

# Step 7: Learn Morfessor segmentation (if enabled)
if [[ "$USE_MORFESSOR" == "true" ]]; then
  echo "Step 7a: Learning Morfessor segmentation on training data..."

  # Check if training files are not empty
  if [[ "$USE_RNN_TAGGER" == "true" ]]; then
    TRAIN_CHECK_SRC="$RNN_DIR/train.rnn.true.$src"
    TRAIN_CHECK_TGT="$RNN_DIR/train.rnn.true.$tgt"
  else
    TRAIN_CHECK_SRC="$MOSES_DIR/train.$src"
    TRAIN_CHECK_TGT="$MOSES_DIR/train.$tgt"
  fi

  if [ ! -s "$TRAIN_CHECK_SRC" ] || [ ! -s "$TRAIN_CHECK_TGT" ]; then
    echo "Error: Training files are empty! Check the preprocessing steps."
    exit 1
  fi

  # Learn Morfessor models for each language separately
  for lang in $src $tgt; do
    echo "Training Morfessor model for $lang..."

    # Choose training data based on --mono flag
    if [[ "$USE_MONO" == "true" && "$lang" == "hsb" ]]; then
      # Use monolingual data for Sorbian language only
      if [[ "$USE_RNN_TAGGER" == "true" ]]; then
        TRAINING_DATA="$mono_src_rnn"
      else
        TRAINING_DATA="$mono_src"
      fi
      echo "Using monolingual data for $lang Morfessor training"
    else
      # Use regular training data (RNN-tagged and truecased if available)
      if [[ "$USE_RNN_TAGGER" == "true" ]]; then
        TRAINING_DATA="$RNN_DIR/train.rnn.true.$lang"
      else
        TRAINING_DATA="$MOSES_DIR/train.$lang"
      fi
      echo "Using training data for $lang Morfessor training"
    fi

    morfessor-train \
      -s "$MORFESSOR_DIR/morfessor_model.$lang.bin" \
      -d ones \
      fi
    "$TRAINING_DATA"
  done

  # Check if Morfessor models were created successfully
  for lang in $src $tgt; do
    if [ ! -f "$MORFESSOR_DIR/morfessor_model.$lang.bin" ]; then
      echo "Error: Morfessor model for $lang was not created!"
      exit 1
    fi
  done

  echo "Morfessor models trained successfully for both languages"

  # Apply Morfessor segmentation to all splits
  echo "Step 7b: Applying Morfessor segmentation to all splits..."
  for split in train dev test; do
    for lang in $src $tgt; do
      echo "Applying Morfessor segmentation to $split.$lang..."

      # Choose input file based on RNN-Tagger usage
      if [[ "$USE_RNN_TAGGER" == "true" ]]; then
        INPUT_FILE="$RNN_DIR/$split.rnn.true.$lang"
        OUTPUT_FILE="$MORFESSOR_DIR/$split.rnn.true.morfessor.$lang"
      else
        INPUT_FILE="$MOSES_DIR/$split.$lang"
        OUTPUT_FILE="$MORFESSOR_DIR/$split.morfessor.$lang"
      fi

      morfessor-segment \
        -l "$MORFESSOR_DIR/morfessor_model.$lang.bin" \
        --output-format-separator "@@ " \
        --output-newlines \
        --output-format "{analysis} " \
        "$INPUT_FILE" \
        >"$OUTPUT_FILE"
    done
  done
  MORFESSOR_INPUT_DIR="$MORFESSOR_DIR"
  if [[ "$USE_RNN_TAGGER" == "true" ]]; then
    MORFESSOR_EXT="rnn.true.morfessor"
  else
    MORFESSOR_EXT="morfessor"
  fi
else
  # No Morfessor preprocessing, use truecased output or Moses output directly
  MORFESSOR_INPUT_DIR="$TRUECASED_INPUT_DIR"
  MORFESSOR_EXT="$TRUECASED_EXT"
fi

# Step 8: Learn BPE on training data (always applied)
echo "Step 8: Learning BPE on training data..."

# Check if training files are not empty
if [[ "$USE_MORFESSOR" == "true" ]]; then
  if [[ "$USE_RNN_TAGGER" == "true" ]]; then
    TRAIN_SRC_FILE="$MORFESSOR_DIR/train.rnn.true.morfessor.$src"
    TRAIN_TGT_FILE="$MORFESSOR_DIR/train.rnn.true.morfessor.$tgt"
  else
    TRAIN_SRC_FILE="$MORFESSOR_DIR/train.morfessor.$src"
    TRAIN_TGT_FILE="$MORFESSOR_DIR/train.morfessor.$tgt"
  fi
elif [[ "$USE_RNN_TAGGER" == "true" ]]; then
  TRAIN_SRC_FILE="$RNN_DIR/train.rnn.true.$src"
  TRAIN_TGT_FILE="$RNN_DIR/train.rnn.true.$tgt"
else
  TRAIN_SRC_FILE="$MOSES_DIR/train.$src"
  TRAIN_TGT_FILE="$MOSES_DIR/train.$tgt"
fi

if [ ! -s "$TRAIN_SRC_FILE" ] || [ ! -s "$TRAIN_TGT_FILE" ]; then
  echo "Error: Training files are empty! Check the preprocessing steps."
  exit 1
fi

# Combine training data and learn BPE
if [[ "$USE_MONO" == "true" ]]; then
  # Use monolingual data for BPE training
  echo "Using monolingual data for BPE training"
  # Always combine mono.hsb with German training data
  if [[ "$DIRECTION" == "hsb-de" ]]; then
    # For hsb->de: mono.hsb + German training data
    if [[ "$USE_RNN_TAGGER" == "true" ]]; then
      cat "$mono_src_rnn" "$TRAIN_TGT_FILE" >"$BPE_DIR/train_combined.txt"
    else
      cat "$mono_src" "$TRAIN_TGT_FILE" >"$BPE_DIR/train_combined.txt"
    fi
  else
    # For de->hsb: mono.hsb + German training data
    if [[ "$USE_RNN_TAGGER" == "true" ]]; then
      cat "$mono_src_rnn" "$TRAIN_SRC_FILE" >"$BPE_DIR/train_combined.txt"
    else
      cat "$mono_src" "$TRAIN_SRC_FILE" >"$BPE_DIR/train_combined.txt"
    fi
  fi
  echo "Combined data (monolingual Sorbian + German training) has $(wc -l <"$BPE_DIR/train_combined.txt") lines"
else
  # Use regular training data
  echo "Using training data for BPE training"
  cat "$TRAIN_SRC_FILE" "$TRAIN_TGT_FILE" >"$BPE_DIR/train_combined.txt"
  echo "Combined training data has $(wc -l <"$BPE_DIR/train_combined.txt") lines"
fi

# Learn BPE with minimum frequency threshold
subword-nmt learn-bpe -s $BPE_OPERATIONS --min-frequency 2 <"$BPE_DIR/train_combined.txt" >"$BPE_DIR/bpe.codes"

# Check if BPE codes were created successfully
if [ ! -s "$BPE_DIR/bpe.codes" ]; then
  echo "Error: BPE codes file is empty or was not created!"
  echo "This might happen with very small datasets. Try reducing BPE_OPERATIONS."
  exit 1
fi

echo "BPE codes learned successfully ($(wc -l <"$BPE_DIR/bpe.codes") operations)"

# Step 9: Apply BPE to all splits
echo "Step 9: Applying BPE to all splits..."

for split in train dev test; do
  for lang in $src $tgt; do
    if [[ "$USE_MORFESSOR" == "true" ]]; then
      if [[ "$USE_RNN_TAGGER" == "true" ]]; then
        INPUT_FILE="$MORFESSOR_DIR/$split.rnn.true.morfessor.$lang"
        OUTPUT_FILE="$BPE_DIR/$split.rnn.true.morfessor.bpe.$lang"
      else
        INPUT_FILE="$MORFESSOR_DIR/$split.morfessor.$lang"
        OUTPUT_FILE="$BPE_DIR/$split.morfessor.bpe.$lang"
      fi
    elif [[ "$USE_RNN_TAGGER" == "true" ]]; then
      INPUT_FILE="$RNN_DIR/$split.rnn.true.$lang"
      OUTPUT_FILE="$BPE_DIR/$split.rnn.true.bpe.$lang"
    else
      INPUT_FILE="$MOSES_DIR/$split.$lang"
      OUTPUT_FILE="$BPE_DIR/$split.bpe.$lang"
    fi
    subword-nmt apply-bpe -c "$BPE_DIR/bpe.codes" <"$INPUT_FILE" >"$OUTPUT_FILE"
  done
done

# Set file extension for fairseq preprocessing
if [[ "$USE_MORFESSOR" == "true" && "$USE_RNN_TAGGER" == "true" ]]; then
  FINAL_EXT="rnn.true.morfessor.bpe"
elif [[ "$USE_MORFESSOR" == "true" ]]; then
  FINAL_EXT="morfessor.bpe"
elif [[ "$USE_RNN_TAGGER" == "true" ]]; then
  FINAL_EXT="rnn.true.bpe"
else
  FINAL_EXT="bpe"
fi

# Step 10: Create fairseq binary dataset
echo "Step 10: Creating fairseq binary dataset..."
fairseq-preprocess \
  --source-lang $src --target-lang $tgt \
  --trainpref "$BPE_DIR/train.$FINAL_EXT" \
  --validpref "$BPE_DIR/dev.$FINAL_EXT" \
  --testpref "$BPE_DIR/test.$FINAL_EXT" \
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
if [[ "$USE_RNN_TAGGER" == "true" ]]; then
  echo "RNN-Tagger: enabled"
fi
if [[ "$USE_MORFESSOR" == "true" ]]; then
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
if [[ "$USE_RNN_TAGGER" == "true" ]]; then
  echo "$RNN_DIR/ - RNN-Tagger output files"
fi
if [[ "$USE_MORFESSOR" == "true" ]]; then
  echo "$MORFESSOR_DIR/ - Morfessor models and segmented files"
fi
echo "$BPE_DIR/ - BPE codes and final segmented files"
echo "$DATA_BIN_DIR/ - Fairseq binary dataset (ready for training)"
echo ""
echo "=== Key Files for NMT Training ==="
echo "Fairseq binary data: $DATA_BIN_DIR/"
echo "BPE codes: $BPE_DIR/bpe.codes"
if [[ "$USE_RNN_TAGGER" == "true" ]]; then
  echo "RNN-Tagger scripts: ./RNNTagger/cmd/rnn-tagger-*.sh"
fi
if [[ "$USE_MORFESSOR" == "true" ]]; then
  echo "Morfessor models: $MORFESSOR_DIR/morfessor_model.$src.bin, $MORFESSOR_DIR/morfessor_model.$tgt.bin"
fi
echo "Truecaser models: $MOSES_DIR/truecase-model.$src, $MOSES_DIR/truecase-model.$tgt"
echo ""
if [[ "$USE_RNN_TAGGER" == "true" && "$USE_MORFESSOR" == "true" ]]; then
  echo "Ready for NMT training with RNN-Tagger + Morfessor + BPE segmentation!"
elif [[ "$USE_RNN_TAGGER" == "true" ]]; then
  echo "Ready for NMT training with RNN-Tagger + BPE segmentation!"
elif [[ "$USE_MORFESSOR" == "true" ]]; then
  echo "Ready for NMT training with Morfessor + BPE segmentation!"
else
  echo "Ready for NMT training with BPE segmentation!"
fi
