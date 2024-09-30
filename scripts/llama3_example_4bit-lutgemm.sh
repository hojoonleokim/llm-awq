MODEL=Meta-Llama-3-8B

# generate real quantized weights (w3)
CUDA_VISIBLE_DEVICES=1 python -m awq.entry --model_path ../model/$MODEL \
    --w_bit 3 --q_group_size 128 \
    --load_awq awq_cache/$MODEL-w3-g128.pt \
    --q_backend real --dump_quant quant_cache/$MODEL-w3-g128-awq-lutgemm.pt
