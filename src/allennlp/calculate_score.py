#! /usr/bin/env python3
import argparse

SHARED_DIR: str = "/projects/instr/19sp/cse481n/GatesNLP/"

def main():
    parser = argparse.ArgumentParser(description='Arguments to be passed into the evaluation script.')
    parser.add_argument('model', type=str, help='the model to evaluate')
    args = parser.parse_args()
    partial = open(SHARED_DIR + "supervised_pairs/rankings_" + args.model + ".txt", 'r').readlines()
    scores = []
    for i in range(2, len(partial), 45):
        scores.append(1/int(partial[i].split()[-1]))
    print(len(scores))
    print(sum(scores) / len(scores))


if __name__ == '__main__':
    main()
