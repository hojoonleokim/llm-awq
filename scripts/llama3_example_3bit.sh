MODEL=Phi-3-medium-4k-instruct
MODEL=Meta-Llama-3-8B-Instruct

# run AWQ search (optional; we provided the pre-computed results)
python -m awq.entry --model_path ../models/$MODEL \
    --w_bit ../index/$MODEL-zscore.pt --q_group_size 128 \
    --run_awq --dump_awq awq_cache/$MODEL-w3.5-g128.pt

# evaluate the AWQ quantize model (simulated pseudo quantization)
python -m awq.entry --model_path ../models/$MODEL \
    --tasks wikitext \
    --w_bit ../index/$MODEL-zscore.pt --q_group_size 128 \
    --load_awq awq_cache/$MODEL-w3.5-g128.pt \
    --q_backend fake
