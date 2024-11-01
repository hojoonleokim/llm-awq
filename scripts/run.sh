MODEL=Meta-Llama-3-8B-Instruct 
WBIT=2
CUDA_VISIBLE_DEVICES=1 python -m awq.entry --model_path ../models/$MODEL \
    --w_bit $WBIT --q_group_size 128 \
    --run_awq --dump_awq awq_cache/$MODEL-w$WBIT-g128.pt

MODEL=Phi-3-medium-4k-instruct  
WBIT=2
CUDA_VISIBLE_DEVICES=1 python -m awq.entry --model_path ../models/$MODEL \
    --w_bit $WBIT --q_group_size 128 \
    --run_awq --dump_awq awq_cache/$MODEL-w$WBIT-g128.pt

# evaluate the AWQ quantize model (simulated pseudo quantization)
MODEL=Meta-Llama-3-8B-Instruct 
for WBIT in 2
do
    for LAYER in {0..31}
    do
        CUDA_VISIBLE_DEVICES=1 python -m awq.entry --model_path ../models/$MODEL \
            --tasks wikitext \
            --w_bit $WBIT --q_group_size 128 -—layer $LAYER \
            --load_awq awq_cache/$MODEL-w$WBIT-g128.pt \
            --q_backend fake
    done
done

MODEL=Phi-3-medium-4k-instruct
for WBIT in 2
do
    for LAYER in {0..39}
    do
        CUDA_VISIBLE_DEVICES=1 python -m awq.entry --model_path ../models/$MODEL \
            --tasks wikitext \
            --w_bit $WBIT --q_group_size 128 -—layer $LAYER \
            --load_awq awq_cache/$MODEL-w$WBIT-g128.pt \
            --q_backend fake
    done
done