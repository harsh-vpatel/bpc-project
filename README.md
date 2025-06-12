# BPC Project

## How to run dataset Preprocessing

### First time

```bash
git clone git@github.com:nilsimda/bpc-project.git
uv venv --python 3.10 # seems to work better with fairseq
uv pip install subword-nmt
uv pip install fairseq
source .venv/bin/activate
chmod +x prepare.sh && ./prepare.sh
```

### Afterwards

```bash
./prepare.sh
```
