import os
import time
import json
import csv
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForMultipleChoice, pipeline
from datasets import load_dataset

script_dir = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    parser = argparse.ArgumentParser(description="Prediciotn using two models on a Question Answering task")
    parser.add_argument(
        "--context_file", 
        type=str, 
        default=None, 
        help="A json file containing the context data.",
        required=True
    )
    parser.add_argument(
        "--test_file", 
        type=str, 
        default=None, 
        help="A json file containing the Prediction data.",
        required=True
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Where to store the prediction results .",
        required=True,
    )
    parser.add_argument(
        "--mc_model_path",
        type=str,
        default="../models_tokenizers_and_data/mc_model/",
        help="Path to mutiple choice model",
        required=False,
    )
    parser.add_argument(
        "--qa_model_path",
        type=str,
        default="../models_tokenizers_and_data/qa_model/",
        help="Path to qa model",
        required=False,
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    
    # check if prediciton.csv already exist
    prediction_csv_path = args.output_csv
    if os.path.isfile(prediction_csv_path):
            raise FileNotFoundError(f"File '{prediction_csv_path}' already exists. Please choose a different file name or location.")

    # load context
    with open(args.context_file, 'r') as file:
        context = json.load(file)

    # load test data
    data_files = {}
    data_files["test"] = args.test_file
    extension = data_files["test"].split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)

    # load mutiple choice model and set to gpu or cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")

    mc_model_path = os.path.join(script_dir, args.mc_model_path)
    tokenizer_mc = AutoTokenizer.from_pretrained(mc_model_path)
    model_mc = AutoModelForMultipleChoice.from_pretrained(mc_model_path)
    model_mc.to(device)

    # use pipeline for qa 
    qa_model_path = os.path.join(script_dir, args.qa_model_path)
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        print(f"Using GPU: {torch.cuda.get_device_name(device_index)}")
    else:
        device_index = None
    question_answerer = pipeline("question-answering", 
                                 model=qa_model_path, 
                                 device=device_index,
                                 Padding=True, 
                                 truncation=True,
                                 doc_stride=128,
                                 max_seq_len=512
                                 )

    # inference
    num_test_data =len(raw_datasets["test"])
    for index in range(num_test_data):
        # test data id, quetion, pragraph candidates
        id = raw_datasets["test"][index]["id"]
        question = raw_datasets["test"][index]["question"]
        context1 = context[raw_datasets["test"][index]['paragraphs'][0]]
        context2 = context[raw_datasets["test"][index]['paragraphs'][1]]
        context3 = context[raw_datasets["test"][index]['paragraphs'][2]]
        context4 = context[raw_datasets["test"][index]['paragraphs'][3]]

        # mc_model inference
        inputs = tokenizer_mc([[question, context1], [question, context2], [question, context3], [question, context4]],
                            return_tensors="pt",
                            padding=True,
                            max_length=512,
                            truncation=True)
        inputs = inputs.to(device)
        labels = torch.tensor(0).unsqueeze(0)
        labels = labels.to(device)
        outputs = model_mc(**{k: v.unsqueeze(0) for k, v in inputs.items()}, labels=labels)
        logits = outputs.logits
        predicted_class = logits.argmax().item()
        
        # qa_model inference
        relevant_context = context[raw_datasets["test"][index]['paragraphs'][predicted_class]]
        answer = question_answerer(question=question, context=relevant_context)["answer"]

        result = {"id": id, "answer": answer}

        # wirte the predicted result to csv file
        with open(prediction_csv_path, mode='a', newline='') as csv_file:
            fieldnames = result.keys()
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            if csv_file.tell() == 0:
                writer.writeheader()

            writer.writerow(result)

        print(f"Prediction {index+1}/{num_test_data}: {result}")

    print("Prediction finished")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time: {elapsed_time} s")
