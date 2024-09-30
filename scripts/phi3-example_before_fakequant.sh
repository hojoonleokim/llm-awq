MODEL=Phi-3-medium-4k-instruct

CUDA_VISIBLE_DEVICES=1 python -m awq.entry --model_path ../model/$MODEL \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 \
    --load_awq awq_cache/$MODEL-w4-g128.pt \
    --q_backend fake --dump_fake ../dumpfake/$MODEL-w4-g128-awq-fake.pt

# load and evaluate the real quantized model (smaller gpu memory usage)
#CUDA_VISIBLE_DEVICES=1 python -m awq.entry --model_path $MODEL \
#    --tasks wikitext \
#    --w_bit 4 --q_group_size 128 \
#    --load_quant quant_cache/$MODEL-w4-g128-awq.pt