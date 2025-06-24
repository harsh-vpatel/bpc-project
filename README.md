# BPC Project

## Install

```bash
git clone git@github.com:nilsimda/bpc-project.git && cd bpc-project
uv sync
source .venv/bin/activate
```

## Prepare dataset

```bash
./prepare.sh [morfessor|bpe]
```

## Evaluate a trained model

```bash

./eval.sh                                                       # Default: BPE model, test set
./eval.sh sorbian_german_morfessor ./dataset/fairseq_morfessor/ # Morfessor model, test set
./eval.sh sorbian_german_bpe ./dataset/fairseq_bpe/ dev         # BPE model, dev set
```
