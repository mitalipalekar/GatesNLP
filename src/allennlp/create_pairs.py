#! /usr/bin/env python3

# Script to sample pairs from a given train, dev, and test set of individual papers, and write
# the sampled pairs to disk

import json
from tqdm import tqdm
from gnlputils import read_dataset
import random

# total count of each type of pair to be split across train/dev/test
CITED_PAIRS = 60000
ONE_HOP = 0
UNCITED_PAIRS = 60000

# train/dev/test split
TRAIN_PERCENT = 0.8
DEV_PERCENT = 0.1
TEST_PERCENT = 0.1

# file paths
SHARED_DIR: str = "/projects/instr/19sp/cse481n/GatesNLP/"
TRAIN: str = SHARED_DIR + 'train.txt'
DEV: str = SHARED_DIR + 'dev.txt'
TEST: str = SHARED_DIR + 'test.txt'
TRAIN_OUTPUT: str = SHARED_DIR + 'supervised_pairs/large_pairs_train.txt'
DEV_OUTPUT: str = SHARED_DIR + 'supervised_pairs/large_pairs_dev.txt'
TEST_OUTPUT: str = SHARED_DIR + 'supervised_pairs/large_pairs_test.txt'


def main():
    # generate pairs
    train_text, train_citations = read_dataset(TRAIN)
    dev_text, dev_citations = read_dataset(DEV)
    test_text, test_citations = read_dataset(TEST)

    train_cited = generate_cited_pairs(train_text, train_citations)
    dev_cited = generate_cited_pairs_across_datasets(dev_text, train_text, dev_citations)
    test_cited = generate_cited_pairs_across_datasets(test_text, train_text, test_citations)

    train_uncited = generate_uncited_pairs(train_text, train_citations, int(UNCITED_PAIRS * TRAIN_PERCENT))
    dev_uncited = generate_uncited_pairs_across_datasets(dev_text, train_text, dev_citations,
                                                         int(UNCITED_PAIRS * DEV_PERCENT))
    test_uncited = generate_uncited_pairs_across_datasets(test_text, train_text, test_citations,
                                                          int(UNCITED_PAIRS * TEST_PERCENT))

    # sample the results down to the desired size
    train_cited = random.sample(train_cited, int(CITED_PAIRS * TRAIN_PERCENT))
    dev_cited = random.sample(dev_cited, int(CITED_PAIRS * DEV_PERCENT))
    test_cited = random.sample(test_cited, int(CITED_PAIRS * TEST_PERCENT))

    train_hops = random.sample(generate_one_hops(train_text, train_citations), int(ONE_HOP * TRAIN_PERCENT))
    dev_hops = random.sample(generate_one_hops_across_datasets(dev_text, train_text, dev_citations, train_citations),
                             int(ONE_HOP * DEV_PERCENT))
    test_hops = random.sample(generate_one_hops_across_datasets(test_text, train_text, test_citations, train_citations),
                              int(ONE_HOP * TEST_PERCENT))

    # write pairs to disk
    write_output(TRAIN_OUTPUT, train_cited + train_hops + train_uncited)
    write_output(DEV_OUTPUT, dev_cited + dev_hops + dev_uncited)
    write_output(TEST_OUTPUT, test_cited + test_hops + test_uncited)


# generate all "one-hop" pairs within text
def generate_one_hops(text, out_citations):
    return generate_one_hops_across_datasets(text, text, out_citations, out_citations)


# generate all "one-hop" pairs consisting of source_text papers citing papers in dest_text citing papers in either
# source or dest
def generate_one_hops_across_datasets(source_text, dest_text, source_out_citations, dest_out_citations):
    one_hops = []
    for paper1 in source_text.keys():
        for paper2 in source_out_citations[paper1]:
            if paper2 in dest_text.keys():
                for paper3 in dest_out_citations[paper2]:
                    if paper3 in source_text.keys():
                        paper3_text = source_text[paper3]
                    elif paper3 in dest_text.keys():
                        paper3_text = dest_text[paper3]
                    if (paper3 in source_text.keys() or paper3 in dest_text.keys()) and paper3 not in source_out_citations[paper1]:
                        result = {}
                        result["query_paper"] = source_text[paper1]
                        result["candidate_paper"] = paper3_text
                        result["relevance"] = "0"
                        one_hops.append(result)
    return one_hops


# generate all cited pairs between papers in text
def generate_cited_pairs(text, out_citations):
    return generate_cited_pairs_across_datasets(text, text, out_citations)


# generate all cited pairs consisting of source_text papers citing papers in dest_text
def generate_cited_pairs_across_datasets(source_text, dest_text, out_citations):
    cited_pairs = []
    for paper1, text1 in source_text.items():
        for paper2 in out_citations[paper1]:
            if paper2 in dest_text.keys():
                result = {}
                result["query_paper"] = source_text[paper1]
                result["candidate_paper"] = dest_text[paper2]
                result["relevance"] = "1"
                cited_pairs.append(result)
    return cited_pairs


# generate pair_count many random cited pairs consisting of source_text papers citing papers in dest_text
def generate_random_cited_pairs(source_text, dest_text, out_citations, pair_count):
    cited_pairs = []
    ids = list(source_text.keys())
    while len(cited_pairs) < pair_count:
        paper1 = random.choice(ids)
        if len(out_citations[paper1]) != 0:
            paper2 = random.choice(out_citations[paper1])
            if paper2 in dest_text.keys():
                result = {}
                result["query_paper"] = source_text[paper1]
                result["candidate_paper"] = dest_text[paper2]
                result["relevance"] = "1"
                cited_pairs.append(result)
    return cited_pairs


# generate pair_count many random uncited pairs from text
def generate_uncited_pairs(text, out_citations, pair_count):
    return generate_uncited_pairs_across_datasets(text, text, out_citations, pair_count)


# generate pair_count many random uncited pairs consisting of source_text papers not citing papers in dest_text
def generate_uncited_pairs_across_datasets(source_text, dest_text, out_citations, pair_count):
    source_ids = list(source_text.keys())
    dest_ids = list(dest_text.keys())
    uncited_pairs = []
    seen = set()
    while len(uncited_pairs) < pair_count:
        paper1 = random.choice(source_ids)
        sampled = False
        while not sampled:
            paper2 = random.choice(dest_ids)
            if paper2 not in out_citations[paper1] and not paper1 == paper2 and (paper1, paper2) not in seen:
                result = {}
                result["query_paper"] = source_text[paper1]
                result["candidate_paper"] = dest_text[paper2]
                result["relevance"] = "0"
                uncited_pairs.append(result)
                seen.add((paper1, paper2))
                sampled = True
    return uncited_pairs


# writes pairs to disk
def write_output(filename, pairs):
    out = open(filename, 'w')
    # shuffle data because we saw empirical improvements with it.
    random.shuffle(pairs)
    for pair in pairs:
        out.write(json.dumps(pair) + '\n')


if __name__ == '__main__':
    main()