#!/bin/bash

input_jsonl="$1"
prediction_jsonl="$2"

echo "Input jsonl: $input_jsonl"
echo "Prediction josnl: $prediction_jsonl"

python code_and_script/prediction.py \
  --model_name_or_path models_tokenizers_and_data/model \
  --train_file $input_jsonl  \
  --validation_file $input_jsonl \
  --source_prefix "summarize: " \
  --text_column maintext \
  --summary_column id \
  --per_device_eval_batch_size 4 \
  --prediction_jsonl $prediction_jsonl \
  --max_train_steps 0 \