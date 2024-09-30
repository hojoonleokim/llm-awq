MODEL=Phi-3-medium-4k-instruct

# generate real quantized weights (w4)
CUDA_VISIBLE_DEVICES=1 python -m awq.entry --model_path ../model/$MODEL \
    --w_bit 3 --q_group_size 128 \
    --load_awq awq_cache/$MODEL-w3-g128.pt \
    --q_backend real --dump_quant quant_cache/$MODEL-w3-g128-awq.pt

# load and evaluate the real quantized model (smaller gpu memory usage)
#CUDA_VISIBLE_DEVICES=1 python -m awq.entry --model_path $MODEL \
#    --tasks wikitext \
#    --w_bit 4 --q_group_size 128 \
#    --load_quant quant_cache/$MODEL-w4-g128-awq.pt