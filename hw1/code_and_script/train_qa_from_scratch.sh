#!/bin/bash

context_json="$1"
train_json="$2"
validation_json="$3"
model_dir="$4"

python run_qa_no_trainer.py \
  --context_file $context_json \
  --train_file $train_json \
  --validation_file $validation_json \
  --tokenizer_name hfl/chinese-roberta-wwm-ext-large \
  --model_type bert \
  --max_seq_length 512 \
  --doc_stride 128 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 2 \
  --learning_rate 0.00003 \
  --with_tracking \
  --output_dir $model_dir