import argparse
import json
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FormatStrFormatter

script_dir = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    parser = argparse.ArgumentParser(description="Plotting report figures")
    parser.add_argument(
        "--log_file", 
        type=str, 
        default='../models_tokenizers_and_data/model/log.txt', 
        help="A log file containing the loss and EM value.",
        required=False,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../models_tokenizers_and_data/model/figures/",
        help="Where to store the figures .",
        required=False,
    )
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    rouge1 = []
    rouge2 = []
    rougel = []
    step = []

    log_file = os.path.join(script_dir, args.log_file)
    output_dir = os.path.join(script_dir, args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            for index, line in enumerate(f):
                if index % 8 == 0:
                    data = json.loads(line)
                    if 'rouge-1' in data and 'f' in data['rouge-1']:
                        rouge1.append(data['rouge-1']['f'])
                    if 'rouge-2' in data and 'f' in data['rouge-2']:
                        rouge2.append(data['rouge-2']['f'])
                    if 'rouge-l' in data and 'f' in data['rouge-l']:
                        rougel.append(data['rouge-l']['f'])
                    if 'step' in data:
                        step.append(data['step'])
                    
    else:
        print(f"File {log_file} not found in the current working directory.")

    # length = len(step)  # set the desired length of the list

    plt.plot(step, rouge1, label='rouge-1')
    plt.plot(step, rouge2, label='rouge-2')
    plt.plot(step, rougel, label='rouge-l')
    plt.xlabel('Step')
    plt.ylabel('Rouge score')

    # plt.gca().yaxis.set_major_formatter(FormatStrFormatter(''))

    # for i in range(0, len(step), 20):
    #     plt.scatter(step[i], train_loss[i], color='b')
    #     plt.annotate(f"({train_loss[i]:.2f})", (step[i]-250, train_loss[i]-0.3), color='b')

    #     plt.scatter(step[i], val_loss[i], color='orange')
    #     plt.annotate(f"({val_loss[i]:.2f})", (step[i]-250, val_loss[i]+0.2), color='orange')

    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'rouge_score.png'))

    max_val = max(rouge1)
    index = rouge1.index(max_val)
    print(f"Max rouge-1 score: {max_val} at step {step[index]}")
    
    max_val = max(rouge2)
    index = rouge2.index(max_val)
    print(f"Max rouge-2 score: {max_val} at step {step[index]}")
    
    max_val = max(rougel)
    index = rougel.index(max_val)
    print(f"Max rouge-l score: {max_val} at step {step[index]}")

if __name__ == "__main__":
    main()