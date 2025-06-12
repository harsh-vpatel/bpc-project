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
./prepare.sh
```
