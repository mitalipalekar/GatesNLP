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
        if id == "78a4d1f70807f8a8b11a82636e8132c655d991cb" or id == "f80c1a392e20a35633a15d718a24ea6f54d2c58a":
            print(text[id])

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
    negative_examples = 10000

    # cited pairs
    processed = 0
    first_paper = ""
    for paper1, text1 in tqdm(text.items()):
        for paper2 in outCitations[paper1]:
            if processed == 15:
                first_paper = paper1
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
                if paper1 == first_paper:
                    print(result)
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
                        if paper1 == first_paper:
                            print(result)
                        out.write(json.dumps(result) + '\n')

    processed = 0
    ids = list(text.keys())
    for paper1, text1 in tqdm(text.items()):
        count = 0
        while count < 5:
            paper2 = random.choice(ids)
            if processed >= negative_examples:
                break
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
                result["query_paper"] = text1
                result["candidate_paper"] = text[paper2]
                result["relevance"] = "1"
                if paper1 == first_paper:
                    print(result)
                out.write(json.dumps(result) + '\n')


if __name__ == '__main__':
    main()
