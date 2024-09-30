MODEL=Meta-Llama-3-8B-Instruct

python -m awq.entry --model_path ../model/$MODEL \
    --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq awq_cache/$MODEL-w4-g128.pt

python -m awq.entry --model_path ../model/$MODEL \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 \
    --load_awq awq_cache/$MODEL-w4-g128.pt \
    --q_backend fake --dump_fake ../dumpfake/$MODEL-w4-g128-awq-bfq.pt
