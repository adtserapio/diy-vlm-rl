torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    main.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path qwen/Qwen2.5-VL-3B-Instruct \
    --dataset_name leonardPKU/clevr_cogen_a_train \
    --max_prompt_length 8192 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --max_pixels 2359296 \
    --save_total_limit 8 \
    --num_train_epochs 1