MODEL=Meta-Llama-3-8B-Instruct

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m awq.entry --model_path /dataset/models/llama3/$MODEL \
    --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq awq_cache/$MODEL-w4-g128.pt

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m awq.entry --model_path ../model/$MODEL \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 \
    --load_awq awq_cache/$MODEL-w4-g128.pt \
    --q_backend fake --dump_fake ../dumpfake/$MODEL-w4-g128-awq-bfq.pt
