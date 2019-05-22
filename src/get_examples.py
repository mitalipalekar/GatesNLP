#! /usr/bin/env python3

import json

PAIRS: str = "/projects/instr/19sp/cse481n/GatesNLP/supervised_pairs/quadruple_pairs_dev.txt"
PREDICTIONS: str = "/projects/instr/19sp/cse481n/GatesNLP/supervised_pairs/quadrupleGpu"
OUTPUT: str = "/projects/instr/19sp/cse481n/GatesNLP/supervised_pairs/examples.txt"

CORRECT: int = 10
INCORRECT: int = 10


def main():
    correct = 0
    incorrect = 0
    with open(PAIRS, 'rb') as pairs, open(PREDICTIONS, 'rb') as predictions, open(OUTPUT, 'w') as out:
        while correct != CORRECT or incorrect != INCORRECT:
            pair = json.loads(pairs.readline())

            prediction_line = predictions.readline()
            while not prediction_line.startswith(bytes("prediction:", encoding="utf-8")):
                prediction_line = predictions.readline()
            prediction_line = prediction_line[13:].strip()
            print(prediction_line)
            prediction = json.loads(prediction_line)

            if pair['relevance'] == prediction["label"]:
                if correct != CORRECT:
                    correct += 1
                    write_example(out, pair, prediction)
            elif incorrect != INCORRECT:
                incorrect += 1
                write_example(out, pair, prediction)


def write_example(out, pair, prediction):
    example = {}
    example['query_paper'] = pair['query_paper']
    example['candidate_paper'] = pair['candidate_paper']
    example['class_probabilities'] = prediction['class_probabilities']
    example['true_label'] = pair['relevance']
    example['predicted_label'] = prediction["label"]
    out.write(json.dumps(example) + '\n')


if __name__ == '__main__':
    main()