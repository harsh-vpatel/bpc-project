# BPC Project

Bidirectional neural machine translation between Upper Sorbian (hsb) and German (de) using Fairseq.

## Install

```bash
git clone git@github.com:nilsimda/bpc-project.git && cd bpc-project
uv sync
source .venv/bin/activate
```

## Prepare dataset

```bash
./prepare.sh                    # Default: hsb→de, BPE-only
./prepare.sh hsb-de             # Explicit hsb→de direction
./prepare.sh de-hsb             # German→Sorbian direction
./prepare.sh hsb-de morfessor   # With Morfessor segmentation
./prepare.sh de-hsb morfessor   # Reverse direction with Morfessor
./prepare.sh hsb-de --mono      # Sorbian→German direction, using additional monolingual training data for bpe
```

## Evaluate a trained model

```bash

./eval.sh                                                       # Default: BPE model, test set
./eval.sh sorbian_german_morfessor ./dataset/fairseq_morfessor/ # Morfessor model, test set
./eval.sh sorbian_german_bpe ./dataset/fairseq_bpe/ dev         # BPE model, dev set
```
