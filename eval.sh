#!/bin/bash

# Enhanced evaluation script for NMT models
# Usage: ./eval.sh [checkpoint_dir] [dataset_dir] [split] [direction]
# Examples:
#   ./eval.sh                                                    # Use defaults (bpe, test, hsb-de)
#   ./eval.sh sorbian_german_bpe ./dataset/fairseq_bpe_hsb-de/   # hsb->de model
#   ./eval.sh german_sorbian_bpe ./dataset/fairseq_bpe_de-hsb/ test de-hsb # de->hsb model
#   ./eval.sh sorbian_german_bpe ./dataset/fairseq_bpe_hsb-de/ dev hsb-de  # hsb->de on dev set

set -e # Exit on any error

# Default parameters
DEFAULT_CHECKPOINT="sorbian_german_bpe"
DEFAULT_DATASET_DIR="./dataset/fairseq_bpe_hsb-de/"
DEFAULT_SPLIT="test"
DEFAULT_DIRECTION="hsb-de"
MOSES_DIR="dataset/output_moses"

# Parse command line arguments
CHECKPOINT_DIR=${1:-$DEFAULT_CHECKPOINT}
FAIRSEQ_DATA_DIR=${2:-$DEFAULT_DATASET_DIR}
SPLIT=${3:-$DEFAULT_SPLIT}
DIRECTION=${4:-$DEFAULT_DIRECTION}

# Validate direction argument
if [[ "$DIRECTION" != "hsb-de" && "$DIRECTION" != "de-hsb" ]]; then
  echo "Error: Invalid direction '$DIRECTION'. Use 'hsb-de' or 'de-hsb'."
  exit 1
fi

# Set source and target languages based on direction
if [[ "$DIRECTION" == "hsb-de" ]]; then
  SRC_LANG="hsb"
  TGT_LANG="de"
else
  SRC_LANG="de"
  TGT_LANG="hsb"
fi

# Validate paths exist
CHECKPOINT_PATH="checkpoints/$CHECKPOINT_DIR/checkpoint_best.pt"
if [ ! -f "$CHECKPOINT_PATH" ]; then
  echo "Error: Checkpoint not found: $CHECKPOINT_PATH"
  echo "Available checkpoints:"
  ls -1 checkpoints/*/checkpoint_best.pt 2>/dev/null | sed 's|checkpoints/||; s|/checkpoint_best.pt||' || echo "  None found"
  exit 1
fi

if [ ! -d "$FAIRSEQ_DATA_DIR" ]; then
  echo "Error: Dataset directory not found: $FAIRSEQ_DATA_DIR"
  exit 1
fi

# Set reference and source files based on direction
REFERENCES_FILE="$MOSES_DIR/$SPLIT.norm.$TGT_LANG"
SOURCE_FILE="$MOSES_DIR/$SPLIT.norm.$SRC_LANG"

if [ ! -f "$REFERENCES_FILE" ]; then
  echo "Error: Reference file not found: $REFERENCES_FILE"
  echo "Available splits in $MOSES_DIR:"
  ls -1 "$MOSES_DIR"/*.de 2>/dev/null | xargs -n1 basename | sed 's/.de$//' || echo "  None found"
  exit 1
fi

# Create results directory and timestamp
mkdir -p results
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_PREFIX="results/eval_${CHECKPOINT_DIR}_${DIRECTION}_${SPLIT}_${TIMESTAMP}"

# Output files
OUTPUT_FILE="${RESULT_PREFIX}_full.txt"
HYPOTHESES_FILE="${RESULT_PREFIX}_hypotheses.txt"
METRICS_FILE="${RESULT_PREFIX}_metrics.txt"

# Function to log with timestamp
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$OUTPUT_FILE"
}

# Function to save and display metrics
save_metrics() {
  local metric_name="$1"
  local output="$2"

  echo "=== $metric_name ===" | tee -a "$METRICS_FILE"
  echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$METRICS_FILE"
  echo "$output" | tee -a "$METRICS_FILE"
  echo "" | tee -a "$METRICS_FILE"
}

# Start evaluation
log "==============================================="
log "       NMT Model Evaluation Report"
log "==============================================="
log "Checkpoint: $CHECKPOINT_DIR"
log "Direction: $DIRECTION ($SRC_LANG -> $TGT_LANG)"
log "Split: $SPLIT"
log "Timestamp: $TIMESTAMP"
log "Source file: $SOURCE_FILE"
log "Reference file: $REFERENCES_FILE"
log "==============================================="

# Count sentences
SENTENCE_COUNT=$(wc -l <"$REFERENCES_FILE")
log "Evaluating $SENTENCE_COUNT sentences..."

log "Step 1/3: Generating translations with fairseq..."

# Map dev to valid for fairseq
FAIRSEQ_SPLIT=${SPLIT/dev/valid}

# Generate translations
fairseq-generate "$FAIRSEQ_DATA_DIR" \
  --path "$CHECKPOINT_PATH" \
  --remove-bpe \
  --source-lang "$SRC_LANG" \
  --target-lang "$TGT_LANG" \
  --arch transformer_iwslt_de_en \
  --tokenizer moses \
  --gen-subset "$FAIRSEQ_SPLIT" \
  --beam 5 \
  --nbest 1 \
  --batch-size 32 \
  --max-len-a 1.2 \
  --max-len-b 10 | tee "${RESULT_PREFIX}_fairseq_output.txt" | grep "^D-" | sort -n -t'-' -k2 | cut -f 3 >"$HYPOTHESES_FILE"

# Verify generation succeeded
if [ ! -s "$HYPOTHESES_FILE" ]; then
  log "ERROR: Translation generation failed - no hypotheses produced"
  exit 1
fi

GENERATED_COUNT=$(wc -l <"$HYPOTHESES_FILE")
log "Generated $GENERATED_COUNT translations"

if [ "$GENERATED_COUNT" -ne "$SENTENCE_COUNT" ]; then
  log "WARNING: Generated $GENERATED_COUNT translations but expected $SENTENCE_COUNT"
fi

log "Step 2/3: Computing BLEU and chrF scores with sacrebleu..."

# Compute sacrebleu metrics
SACREBLEU_OUTPUT=$(sacrebleu "$REFERENCES_FILE" -i "$HYPOTHESES_FILE" -m bleu chrf --chrf-word-order 2 2>&1)
save_metrics "SacreBLEU Metrics" "$SACREBLEU_OUTPUT"

log "Step 3/3: Computing COMET score..."

# Compute COMET score (with error handling and clean output)
if command -v comet-score &>/dev/null; then
  # Suppress verbose output, capture only final score
  COMET_FULL=$(comet-score -s "$SOURCE_FILE" -t "$HYPOTHESES_FILE" -r "$REFERENCES_FILE" 2>/dev/null || echo "COMET evaluation failed")
  COMET_OUTPUT=$(echo "$COMET_FULL" | grep "score:" | tail -1)

  save_metrics "COMET Score" "$COMET_OUTPUT"
else
  log "WARNING: comet-score not found - skipping COMET evaluation"
  echo "=== COMET Score ===" >>"$METRICS_FILE"
  echo "COMET not installed - install with: pip install unbabel-comet" >>"$METRICS_FILE"
  echo "" >>"$METRICS_FILE"
fi

log "==============================================="
log "Evaluation completed successfully!"
log "==============================================="
log "Results saved to:"
log "  Full log: $OUTPUT_FILE"
log "  Metrics: $METRICS_FILE"
log "  Hypotheses: $HYPOTHESES_FILE"
log "  Fairseq output: ${RESULT_PREFIX}_fairseq_output.txt"
log "==============================================="

# Display quick summary
echo ""
echo "=== QUICK SUMMARY ==="

# Parse JSON metrics using jq if available, fallback to grep
if command -v jq &>/dev/null; then
  BLEU_SCORE=$(echo "$SACREBLEU_OUTPUT" | jq -r '.[] | select(.name=="BLEU") | "BLEU = " + (.score | tostring)')
  CHRF_SCORE=$(echo "$SACREBLEU_OUTPUT" | jq -r '.[] | select(.name=="chrF2++") | "chrF2++ = " + (.score | tostring)')
else
  BLEU_SCORE=$(echo "$SACREBLEU_OUTPUT" | grep -o '"score": [0-9.]*' | head -1 | sed 's/"score": /BLEU = /')
  CHRF_SCORE=$(echo "$SACREBLEU_OUTPUT" | grep -o '"score": [0-9.]*' | tail -1 | sed 's/"score": /chrF2++ = /')
fi

# Extract COMET score
if command -v comet-score &>/dev/null && [[ -n "$COMET_OUTPUT" ]]; then
  COMET_SCORE=$(echo "$COMET_OUTPUT" | grep -o '[0-9.]*$' | head -1)
  COMET_DISPLAY="COMET = $COMET_SCORE"
else
  COMET_DISPLAY="COMET: Not available"
fi

# Display summary
echo "Model: $CHECKPOINT_DIR ($DIRECTION)"
echo "Sentences: $SENTENCE_COUNT"
echo "$BLEU_SCORE"
echo "$CHRF_SCORE"
echo "$COMET_DISPLAY"
echo "======================="
