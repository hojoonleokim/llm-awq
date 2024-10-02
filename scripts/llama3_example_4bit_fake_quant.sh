MODEL=Phi-3-medium-4k-instruct


CUDA_VISIBLE_DEVICES=0 python -m awq.entry --model_path ../models/$MODEL \
    --tasks wikitext \
    --w_bit 3 --q_group_size 128 \
    --load_awq awq_cache/$MODEL-w3-g128.pt \
    --q_backend fake --dump_fake ../fake_cache/$MODEL-w3-g128-fakequant-cliped
