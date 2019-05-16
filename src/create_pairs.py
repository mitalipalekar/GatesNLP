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
    train = open(TRAIN, 'w')
    dev = open(DEV, 'w')
    test = open(TEST, 'w')

    text = dict()
    outCitations = dict()

    for line in f:
        paper = json.loads(line)
        id = paper['id']
        text[id] = paper['title'] + ' ' + paper['paperAbstract']
        outCitations[id] = paper['outCitations']

    true_citation_count = 15000
    one_hop_count = 7500
    negative_examples = 7500

    # random cited pairs
    processed = 0
    ids = list(text.keys())
    while processed < true_citation_count:
        paper1 = random.choice(ids)
        if len(outCitations[paper1]) != 0:
            paper2 = random.choice(outCitations[paper1])
            if paper2 in text.keys():
                if processed < int(0.8 * true_citation_count):
                    out = train
                elif processed >= int(0.8 * true_citation_count) and processed < int(0.9 * true_citation_count):
                    out = dev
                else:
                    out = test
                processed += 1
                result = {}
                result["query_paper"] = text[paper1]
                result["candidate_paper"] = text[paper2]
                result["relevance"] = "1"
                out.write(json.dumps(result) + '\n')
    print(processed)

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
                        if processed < int(0.8 * one_hop_count):
                            out = train
                        elif processed >= int(0.8 * one_hop_count) and processed < int(0.9 * one_hop_count):
                            out = dev
                        else:
                            out = test
                        processed += 1
                        result = {}
                        result["query_paper"] = text[paper1]
                        result["candidate_paper"] = text[paper3]
                        result["relevance"] = "0"
                        out.write(json.dumps(result) + '\n')
    print(processed)

    # random negatives
    processed = 0
    while processed < negative_examples:
        paper1 = random.choice(ids)
        count = 0
        while count < 1:
            paper2 = random.choice(ids)
            if paper2 not in outCitations[paper1] and not paper1 == paper2:
                if processed < int(0.8 * negative_examples):
                    out = train
                elif processed >= int(0.8 * negative_examples) and processed < int(0.9 * negative_examples):
                    out = dev
                else:
                    out = test
                processed += 1
                count += 1
                result = {}
                result["query_paper"] = text[paper1]
                result["candidate_paper"] = text[paper2]
                result["relevance"] = "0"
                out.write(json.dumps(result) + '\n')
    print(processed)


if __name__ == '__main__':
    main()
