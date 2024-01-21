export CUDA_VISIBLE_DEVICES=8

python qlora.py \
    --optim adamw_torch \
    --max_train_samples 5000 \
    --model_name_or_path yentinglin/Taiwan-LLM-7B-v2.0-chat\
    --use_auth \
    --output_dir ./output/v10 \
    --logging_steps 10 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 10 \
    --save_total_limit 40 \
    --evaluation_strategy steps \
    --eval_dataset_size 1024 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 32 \
    --dataloader_num_workers 1 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --lora_r 4 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --fp16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --dataset data/train.json \
    --eval_dataset data/public_test.json \
    --source_max_len 512 \
    --target_max_len 512 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_steps 250 \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 0 \