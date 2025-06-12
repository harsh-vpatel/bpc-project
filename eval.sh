fairseq-generate ./dataset/fairseq \
  --path checkpoints/sorbian_german_transformer/checkpoint_last.pt \
  --batch-size 128 \
  --beam 5 \
  --remove-bpe \
  --source-lang hsb \
  --arch transformer \
  --target-lang de
