#! /usr/bin/env python3
import json
from tqdm import tqdm
import random

PAPERS = '/projects/instr/19sp/cse481n/GatesNLP/extended_dataset.txt'
TRAIN = '/projects/instr/19sp/cse481n/GatesNLP/supervised_pairs/pairs_train.txt'
DEV = '/projects/instr/19sp/cse481n/GatesNLP/supervised_pairs/pairs_dev.txt'
TEST = '/projects/instr/19sp/cse481n/GatesNLP/supervised_pairs/pairs_test.txt'

def main():
    f = open(PAPERS, 'r')

    text = dict()
    outCitations = dict()

    for line in f:
        paper = json.loads(line)
        id = paper['id']
        text[id] = paper['title'] + ' ' + paper['paperAbstract']
        outCitations[id] = paper['outCitations']

    true_citation_count = 60000
    one_hop_count = 0
    negative_examples = 60000

    processed_train = []
    processed_dev = []
    processed_test = []

    # random cited pairs
    processed = 0
    ids = list(text.keys())
    while processed < true_citation_count:
        paper1 = random.choice(ids)
        if len(outCitations[paper1]) != 0:
            paper2 = random.choice(outCitations[paper1])
            if paper2 in text.keys():
                result = {}
                result["query_paper"] = text[paper1]
                result["candidate_paper"] = text[paper2]
                result["relevance"] = "1"
                if processed < int(0.8 * true_citation_count):
                    processed_train.append(result)
                elif processed >= int(0.8 * true_citation_count) and processed < int(0.9 * true_citation_count):
                    processed_dev.append(result)
                else:
                    processed_test.append(result)
                processed += 1

    processed = 0

    # one-hop pairs
    while processed < one_hop_count:
        paper1 = random.choice(ids)
        if len(outCitations[paper1]) != 0:
            paper2 = random.choice(outCitations[paper1])
            if paper2 in text.keys():
                if len(outCitations[paper2]) != 0:
                    paper3 = random.choice(outCitations[paper2])
                    if paper3 in text.keys() and paper3 not in outCitations[paper1]:
                        result = {}
                        result["query_paper"] = text[paper1]
                        result["candidate_paper"] = text[paper3]
                        result["relevance"] = "0"
                        if processed < int(0.8 * one_hop_count):
                            processed_train.append(result)
                        elif processed >= int(0.8 * one_hop_count) and processed < int(0.9 * one_hop_count):
                            processed_dev.append(result)
                        else:
                            processed_test.append(result)
                        processed += 1

    # random negatives
    processed = 0
    while processed < negative_examples:
        paper1 = random.choice(ids)
        count = 0
        while count < 1:
            paper2 = random.choice(ids)
            if paper2 not in outCitations[paper1] and not paper1 == paper2:
                result = {}
                result["query_paper"] = text[paper1]
                result["candidate_paper"] = text[paper2]
                result["relevance"] = "0"
                if processed < int(0.8 * negative_examples):
                    processed_train.append(result)
                elif processed >= int(0.8 * negative_examples) and processed < int(0.9 * negative_examples):
                    processed_dev.append(result)
                else:
                    processed_test.append(result)
                processed += 1
                count += 1


    output_shuffle(open(TRAIN, 'w'), processed_train)
    output_shuffle(open(DEV, 'w'), processed_dev)
    output_shuffle(open(TEST, 'w'), processed_test)


def output_shuffle(out, outputs):
    random.shuffle(outputs)
    for output in outputs:
        out.write(json.dumps(output) + '\n')


if __name__ == '__main__':
    main()
