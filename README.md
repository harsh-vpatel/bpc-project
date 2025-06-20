# BPC Project

## Install

```bash
git clone git@github.com:nilsimda/bpc-project.git && cd bpc-project
uv venv --python 3.10 # seems to work better with fairseq
uv pip install subword-nmt
uv pip install fairseq
source .venv/bin/activate
```

## Prepare dataset

```bash
./prepare.sh [morfessor|bpe]
```

## Evaluate a trained model
```bash
./eval.sh                                    # Default: BPE model, test set
./eval.sh sorbian_german_morfessor morfessor # Morfessor model, test set
./eval.sh sorbian_german_bpe bpe dev         # BPE model, dev set
```
