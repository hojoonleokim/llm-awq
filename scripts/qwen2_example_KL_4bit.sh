MODEL="Qwen/Qwen2.5-32B-Instruct"

for w_bit in 4; do
    for layer in {18..63}; do
        echo "Running with w_bit=$w_bit, layer=$layer"
        
        # evaluate the AWQ quantize model (simulated pseudo quantization)
        CUDA_VISIBLE_DEVICES=2,3 python -m awq.entry --model_path $MODEL \
            --tasks wikitext \
            --w_bit $w_bit --q_group_size 128 \
            --load_awq awq_cache/$MODEL-w${w_bit}-g128.pt \
            --q_backend fake --layer $layer
        
        echo "Completed w_bit=$w_bit, layer=$layer"
        echo "----------------------------------------"
    done
done