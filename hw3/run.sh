#!/bin/bash

twllama_folder="$1"
peft_model_folder="$2"
input_file="$3"
ouput_file="$4"

echo "twllama_folder: $twllama_folder"
echo "peft_model_folder: $peft_model_folder"
echo "input_file: $input_file"
echo "ouput_file: $ouput_file"

python qlora.py \
    --model_name_or_path $twllama_folder \
    --peft_model_dir $peft_model_folder \
    --output_dir $peft_model_folder \
    --do_train False \
    --do_eval False \
    --do_predict True \
    --predict_with_generate \
    --per_device_eval_batch_size 1 \
    --dataset $input_file \
    --prediction_output_file $ouput_file \
    --source_max_len 512 \
    --target_max_len 512 \
    --max_new_tokens 128 \
    --nums_beams 2 \
    --do_sample \
    --top_p 0.2 \
    --temperature 1.2 \
    --seed 42 \
    --gradient_checkpointing=False \