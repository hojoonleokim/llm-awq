MODEL=Phi-3-medium-4k-instruct
MODEL=Meta-Llama-3-8B-Instruct

MODEL=Phi-3-medium-4k-instruct
MODEL=Meta-Llama-3-8B-Instruct

# generate real quantized weights (w3)
python -m awq.entry --model_path ../models/$MODEL \
    --w_bit 4 --q_group_size 128 \
    --load_awq awq_cache/$MODEL-w4-g128.pt \
    --q_backend real --dump_quant quant_cache/$MODEL-w4-g128-lutgemm.pt
