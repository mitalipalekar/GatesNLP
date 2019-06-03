#! /usr/bin/env python3
import argparse

SHARED_DIR: str = "/projects/instr/19sp/cse481n/GatesNLP/"

def main():
    parser = argparse.ArgumentParser(description='Arguments to be passed into the evaluation script.')
    parser.add_argument('-logs', type=str, nargs='+',
                        help='the model from which to calculate the score from its log', default=[])
    parser.add_argument('-scores', type=str, nargs='+',
                        help='the model from which to calculate the score from its scores', default=[])
    args = parser.parse_args()

    score_sum = 0
    score_count = 0
    for model in args.logs:
        log = open(SHARED_DIR + "supervised_pairs/rankings_" + model + ".txt", 'r').readlines()
        for i in range(len(log)):
            if (log[i].startswith("First cited at ")):
                score_count += 1
                score_sum += 1/int(log[i].split()[-1])

    for model in args.scores:
        log = open(SHARED_DIR + "supervised_pairs/rankings_" + model + "_score" + ".txt", 'r').readlines()
        for i in range(len(log)):
            split = log[i].split()
            score_sum = float(split[0])
            score_count = int(split[1])

    print(score_count)
    print(score_sum / score_count)


if __name__ == '__main__':
    main()
