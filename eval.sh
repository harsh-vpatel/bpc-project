CUDA_VISIBLE_DEVICES=1 fairseq-generate ./dataset/fairseq \
    --path checkpoints/sorbian_german_transformer/checkpoint_best.pt \
    --batch-size 128 \
    --beam 5 \
    --remove-bpe \
    --source-lang hsb \
    --target-lang de \
    --arch transformer_iwslt_de_en
