MODEL="Qwen/Qwen2.5-32B-Instruct"

# run AWQ search (optional; we provided the pre-computed results)
CUDA_VISIBLE_DEVICES=3 python -m awq.entry --model_path $MODEL \
    --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq awq_cache/$MODEL-w4-g128.pt

# # evaluate the AWQ quantize model (simulated pseudo quantization)
CUDA_VISIBLE_DEVICES=3 python -m awq.entry --model_path $MODEL \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 \
    --load_awq awq_cache/$MODEL-w4-g128.pt \
    --q_backend fake --dump_fake fakequant_cache/$MODEL-w4-g128-fakequant

CUDA_VISIBLE_DEVICES=3 python -m awq.entry --model_path $MODEL \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 \
    --load_awq awq_cache/$MODEL-w4-g128.pt \
    --q_backend fake --dump_scaled scaled_fp16_cache/$MODEL-w4-g128-scaled

# # generate real quantized weights (w3)
CUDA_VISIBLE_DEVICES=3 python -m awq.entry --model_path $MODEL \
    --w_bit 4 --q_group_size 128 \
    --load_awq awq_cache/$MODEL-w4-g128.pt \
    --q_backend real --dump_quant quant_cache/$MODEL-w4-g128-lutgemm.pt

# load and evaluate the real quantized model (smaller gpu memory usage)
#python -m awq.entry --model_path ../models/$MODEL \
#    --tasks wikitext \
#    --w_bit 3 --q_group_size 128 \
#    --load_quant quant_cache/$MODEL-w3-g128-awq.pt