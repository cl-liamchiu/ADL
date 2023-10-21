#!/bin/bash

context_json="$1"
test_json="$2"
prediction_csv="$3"

echo "Context JSON: $context_json"
echo "Test JSON: $test_json"
echo "Prediction CSV: $prediction_csv"

python code_and_script/prediction.py \
  --context_file $context_json \
  --test_file $test_json \
  --output_csv $prediction_csv \

