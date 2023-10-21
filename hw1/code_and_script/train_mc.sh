#!/bin/bash

context_json="$1"
train_json="$2"
validation_json="$3"
model_dir="$4"

python run_swag_no_trainer.py \
  --context_file $context_json \
  --train_file $train_json \
  --validation_file $validation_json \
  --model_name_or_path hfl/chinese-roberta-wwm-ext \
  --max_seq_length 512 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --num_train_epochs 1 \
  --learning_rate 3e-5 \
  --output_dir $model_dir