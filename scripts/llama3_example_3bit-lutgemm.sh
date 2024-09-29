MODEL=Meta-Llama-3-8B

# run AWQ search (optional; we provided the pre-computed results)
python CUDA_VISIBLE_DEVICES=1 -m awq.entry --model_path ../models/$MODEL \
    --w_bit 3 --q_group_size 128 \
    --run_awq --dump_awq awq_cache/$MODEL-w3-g128.pt

# evaluate the AWQ quantize model (simulated pseudo quantization)
python CUDA_VISIBLE_DEVICES=1 -m awq.entry --model_path ../models/$MODEL \
    --tasks wikitext \
    --w_bit 3 --q_group_size 128 \
    --load_awq awq_cache/$MODEL-w3-g128.pt \
    --q_backend fake

# generate real quantized weights (w3)
python CUDA_VISIBLE_DEVICES=1 -m awq.entry --model_path ../models/$MODEL \
    --w_bit 3 --q_group_size 128 \
    --load_awq awq_cache/$MODEL-w3-g128.pt \
    --q_backend real --dump_quant quant_cache/$MODEL-w3-g128-awq-lutgemm.pt