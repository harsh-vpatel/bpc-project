# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a bidirectional neural machine translation (NMT) project for translating between Upper Sorbian (hsb) and German (de) using Facebook's Fairseq framework. The project supports both translation directions (hsb↔de) and uses a transformer architecture (IWSLT variant) trained on preprocessed parallel text data.

## Development Environment Setup

```bash
# Clone and setup environment
git clone git@github.com:nilsimda/bpc-project.git && cd bpc-project
uv sync  # Installs all dependencies including fairseq, subword-nmt, morfessor
source .venv/bin/activate
```

## Key Commands

### Data Preprocessing
```bash
./prepare.sh                    # Default: hsb→de, BPE-only pipeline
./prepare.sh hsb-de             # Explicit hsb→de direction, BPE-only
./prepare.sh de-hsb             # German→Sorbian direction, BPE-only
./prepare.sh hsb-de morfessor   # hsb→de with Morfessor + BPE pipeline
./prepare.sh de-hsb morfessor   # de→hsb with Morfessor + BPE pipeline
```

### Training
```bash
# BPE training
bash train_bpe_hsb-de.sh

# Morfessor training  
bash train_morfessor_hsb-de.sh
```

### Evaluation
```bash
./eval.sh                                                       # Default: BPE model, test set
./eval.sh sorbian_german_morfessor ./dataset/fairseq_morfessor/ # Morfessor model, test set
./eval.sh sorbian_german_bpe ./dataset/fairseq_bpe/ dev         # BPE model, dev set
```

## Architecture

### Data Processing Pipeline
1. **Moses preprocessing**: Normalize punctuation → Tokenize → Clean → Truecase
2. **Optional Morfessor segmentation**: Apply morphological segmentation (when `morfessor` flag is used)
3. **BPE segmentation**: Always applied (16K operations, min-frequency 2)
4. **Fairseq preprocessing**: Convert to binary format for training

### Directory Structure
- `dataset/original/`: Train/dev/test split files (train.hsb, train.de, etc.)
- `dataset/output_moses/`: Moses preprocessing output (shared across directions)
- `dataset/output_bpe/`: BPE codes and BPE-only segmented files (shared across directions)
- `dataset/output_morfessor/`: Morfessor models and morphologically segmented files (shared across directions)
- `dataset/output_morfessor_bpe/`: BPE codes and files with Morfessor+BPE segmentation (shared across directions)
- `dataset/fairseq_bpe_hsb-de/`: Binary dataset for hsb→de BPE-only training
- `dataset/fairseq_bpe_de-hsb/`: Binary dataset for de→hsb BPE-only training
- `dataset/fairseq_morfessor_bpe_hsb-de/`: Binary dataset for hsb→de Morfessor+BPE training
- `dataset/fairseq_morfessor_bpe_de-hsb/`: Binary dataset for de→hsb Morfessor+BPE training
- `checkpoints/`: Model checkpoints during training
- `results/`: Evaluation results and metrics
- `moses_scripts/`: Perl scripts for text preprocessing

### Model Configuration
- Architecture: `transformer_iwslt_de_en`
- Training: 50 epochs max, early stopping (patience=10)
- Optimization: Adam with inverse sqrt LR schedule
- Evaluation: BLEU score with beam search (beam=5)
- GPU: Single GPU training/evaluation

## Important Notes

- Data files must be pre-split as train/dev/test in `dataset/original/` (train.hsb, train.de, etc.)
- BPE segmentation is always applied (16K operations)
- Morfessor morphological segmentation can be applied as preprocessing before BPE
- BPE operations may need adjustment for very small datasets
- Model uses shared decoder input/output embeddings
- Checkpoints saved every epoch, best model selected by BLEU score
- Uses label smoothed cross-entropy loss with 0.1 smoothing
- Evaluation produces BLEU, chrF2++, and COMET scores with detailed result files

## Dependencies

- Python 3.10+ (required for fairseq compatibility)
- fairseq (Facebook's sequence-to-sequence toolkit)
- subword-nmt (for BPE segmentation)
- morfessor (for morphological segmentation)
- sacremoses (for Moses preprocessing)
- unbabel-comet (for COMET evaluation metric)
- torch (PyTorch, version <2.4 for fairseq compatibility)
