#! /usr/bin/env python3

import json

from gnlputils import split_all_data

SHARED_DIR: str = "/projects/instr/19sp/cse481n/GatesNLP/"
PAPERS: str = SHARED_DIR + "extended_dataset.txt"
TRAIN = SHARED_DIR + 'train.txt'
DEV = SHARED_DIR + "dev.txt"
TEST = SHARED_DIR + 'test.txt'

def main():
    lines = []
    with open(PAPERS, 'rb') as f:
        for line in f:
            lines.append(json.loads(line))

    lines.sort(key=lambda x: x['year'])

    train, dev, test = split_all_data(lines, 0.8, 0.9)
    write_dataset(TRAIN, train)
    write_dataset(DEV, dev)
    write_dataset(TEST, test)


def write_dataset(filename, dataset):
    out = open(filename, 'w')
    for output in dataset:
        out.write(json.dumps(output) + '\n')


if __name__ == '__main__':
    main()