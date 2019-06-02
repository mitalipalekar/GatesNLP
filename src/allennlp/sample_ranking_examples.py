#! /usr/bin/env python3
import argparse
import random
from gnlputils import read_dataset

SHARED_DIR: str = "/projects/instr/19sp/cse481n/GatesNLP/"
TRAIN: str = SHARED_DIR + "train.txt"
DEV: str = SHARED_DIR + "dev.txt"
TEST: str = SHARED_DIR + "test.txt"

def main():
    parser = argparse.ArgumentParser(description='Arguments to be passed into the evaluation script.')
    parser.add_argument('model', type=str, help='the model to sample examples for')
    parser.add_argument('example_count', type=int, help='how many examples to sample')
    parser.add_argument('--test', action='store_true', help='whether to run for test')
    args = parser.parse_args()

    train_texts, _ = read_dataset(TRAIN)
    if args.test:
        eval_texts, _= read_dataset(TEST)
    else:
        eval_texts, _ = read_dataset(DEV)

    log = open(SHARED_DIR + "supervised_pairs/rankings_" + args.model + ('' if args.test else "_dev")
               + ".txt", 'r').readlines()
    starting_indices = []
    ranking_count = 0
    for line_number, line in enumerate(log):
        if line.startswith("RANKING"):
            ranking_count += 1
            starting_indices.append(line_number)
    samples = random.sample(range(ranking_count), args.example_count)
    output = open(SHARED_DIR + "supervised_pairs/samples_" + args.model + ('' if args.test else "_dev") + ".txt", 'w')
    for i in samples:
        log_index = starting_indices[i]
        while log[log_index] != "\n":
            output.write(log[log_index])
            if log[log_index] == "QUERY\n":
                log_index += 1
                output.write(eval_texts[log[log_index].strip()] + "\n")
            elif log[log_index].startswith("RANK") and not log[log_index].startswith("RANKING"):
                curr_line = log[log_index].split()
                if len(curr_line) == 2:
                    log_index += 1
                    output.write(train_texts[log[log_index].strip()] + "\n")
                else:
                    output.write(train_texts[curr_line[-1].strip()] + "\n")
            log_index += 1
        output.write(log[log_index])


if __name__ == '__main__':
    main()
