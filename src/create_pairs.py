#! /usr/bin/env python3
import json
from tqdm import tqdm

PAPERS = '/projects/instr/19sp/cse481n/GatesNLP/extended_dataset.txt'
TRAIN = '/projects/instr/19sp/cse481n/GatesNLP/pairs_train.txt'
DEV = '/projects/instr/19sp/cse481n/GatesNLP/pairs_dev.txt'
TEST = '/projects/instr/19sp/cse481n/GatesNLP/pairs_test.txt'

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

    # counts numbers for train/dev/test split
    true_citation_count = 0
    one_hop_count = 0
    for paper1, text1 in tqdm(text.items()):
        for paper2 in outCitations[paper1]:
            if paper2 in text.keys():
                true_citation_count += 1
    for paper1, text1 in tqdm(text.items()):
        seen = set()
        for paper2 in outCitations[paper1]:
            if paper2 in text.keys():
                for paper3 in outCitations[paper2]:
                    if paper3 in text.keys() and paper3 not in seen and paper3 not in outCitations[paper1]:
                        seen.add(paper3)
                        one_hop_count += 1
    true_citation_count = 10000
    one_hop_count = 10000

    # cited pairs
    processed = 0
    for paper1, text1 in tqdm(text.items()):
        for paper2 in outCitations[paper1]:
            if paper2 in text.keys() and processed < true_citation_count:
                if processed < int(0.8 * true_citation_count):
                    out = train
                elif processed >= int(0.8 * true_citation_count) and processed < int(0.9 * true_citation_count):
                    out = dev
                else:
                    out = test
                processed += 1
                result = {}
                result["query_paper"] = text1
                result["candidate_paper"] = text[paper2]
                result["relevance"] = "1"
                out.write(json.dumps(result) + '\n')

    processed = 0

    # one-hop pairs
    for paper1, text1 in tqdm(text.items()):
        seen = set()
        for paper2 in outCitations[paper1]:
            if paper2 in text.keys():
                for paper3 in outCitations[paper2]:
                    if paper3 in text.keys() and paper3 not in seen and paper3 not in outCitations[paper1] and processed < one_hop_count:
                        if processed < int(0.8 * one_hop_count):
                            out = train
                        elif processed >= int(0.8 * one_hop_count) and processed < int(0.9 * one_hop_count):
                            out = dev
                        else:
                            out = test
                        seen.add(paper3)
                        processed += 1
                        result = {}
                        result["query_paper"] = text1
                        result["candidate_paper"] = text[paper3]
                        result["relevance"] = "0"
                        out.write(json.dumps(result) + '\n')


if __name__ == '__main__':
    main()
