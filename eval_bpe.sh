CUDA_VISIBLE_DEVICES=0 fairseq-generate ./dataset/fairseq_bpe \
  --path checkpoints/sorbian_german_bpe/checkpoint_best.pt \
  --batch-size 128 \
  --beam 5 \
  --remove-bpe \
  --source-lang hsb \
  --target-lang de \
  --arch transformer_iwslt_de_en
