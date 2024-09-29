MODEL=Meta-Llama-3-8B



# evaluate the AWQ quantize model (simulated pseudo quantization)
CUDA_VISIBLE_DEVICES=1 python -m awq.entry --model_path ../model/$MODEL \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 \
    --load_awq awq_cache/$MODEL-w4-g128.pt \
    --q_backend fake --dump_fake "y"
