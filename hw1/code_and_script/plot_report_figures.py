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
        default='../models_tokenizers_and_data/qa_model/log.txt', 
        help="A log file containing the loss and EM value.",
        required=False,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../models_tokenizers_and_data/qa_model/figures/",
        help="Where to store the figures .",
        required=False,
    )
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    train_loss = []
    val_loss = []
    exact_match = []
    step = []

    log_file = os.path.join(script_dir, args.log_file)
    output_dir = os.path.join(script_dir, args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                if 'train_loss' in data:
                    train_loss.append(data['train_loss'])
                if 'val_loss' in data:
                    val_loss.append(data['val_loss'])
                if 'step' in data:
                    if data['step'] not in step:
                        step.append(data['step'])
                if 'Evaluation metrics' in data and 'exact_match' in data['Evaluation metrics']:
                    exact_match.append(data['Evaluation metrics']['exact_match'])
    else:
        print(f"File {log_file} not found in the current working directory.")

    length = len(train_loss)  # set the desired length of the list

    step = list(range(0, length * 50, 50))

    plt.plot(step, train_loss, label='Train Loss', color='skyblue')
    plt.plot(step, val_loss, label='Val Loss', color='#FFA07A')
    plt.xlabel('Step')
    plt.ylabel('Loss')

    plt.gca().yaxis.set_major_formatter(FormatStrFormatter(''))

    for i in range(0, len(step), 20):
        plt.scatter(step[i], train_loss[i], color='b')
        plt.annotate(f"({train_loss[i]:.2f})", (step[i]-250, train_loss[i]-0.3), color='b')

        plt.scatter(step[i], val_loss[i], color='orange')
        plt.annotate(f"({val_loss[i]:.2f})", (step[i]-250, val_loss[i]+0.2), color='orange')

    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss.png'))

    plt.clf()
    plt.plot(step, exact_match, label='Exact Match', color='skyblue')
    plt.xlabel('Step')
    plt.ylabel('Exact Match')

    plt.gca().yaxis.set_major_formatter(FormatStrFormatter(''))

    for i in range(0, len(step), 20):
        plt.scatter(step[i], exact_match[i], color='b')
        plt.annotate(f"({exact_match[i]:.1f})", (step[i]-250, exact_match[i]+2), color='b')

    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'exact_match.png'))

if __name__ == "__main__":
    main()