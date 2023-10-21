#!/bin/bash

context_json="$1"
train_json="$2"
validation_json="$3"
model_dir="$4"

python run_qa_no_trainer_end_to_end.py \
  --context_file $context_json \
  --train_file $train_json \
  --validation_file $validation_json \
  --model_name_or_path schen/longformer-chinese-base-4096 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --per_device_train_batch_size 12 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 2 \
  --learning_rate 0.00003 \
  --with_tracking \
  --output_dir $model_dir