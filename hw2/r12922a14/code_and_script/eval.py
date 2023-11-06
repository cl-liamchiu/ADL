import json
import argparse
from tw_rouge import get_rouge


def main(args):
    refs, preds = {}, {}

    with open(args.reference) as file:
        for line in file:
            line = json.loads(line)
            refs[line['id']] = line['title'].strip() + '\n'

    with open(args.submission) as file:
        for line in file:
            line = json.loads(line)
            preds[line['id']] = line['title'].strip() + '\n'

    keys =  refs.keys()
    refs = [refs[key] for key in keys]
    preds = [preds[key] for key in keys]

    rouge_score = get_rouge(preds, refs)
    print(json.dumps(rouge_score, indent=2))
    print(f'rouge-1: {rouge_score["rouge-1"]["f"]*100: .2f}, rouge-2: {rouge_score["rouge-2"]["f"]*100: .2f}, rouge-l: {rouge_score["rouge-l"]["f"]*100: .2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reference')
    parser.add_argument('-s', '--submission')
    args = parser.parse_args()
    main(args)
