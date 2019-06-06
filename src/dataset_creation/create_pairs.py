#! /usr/bin/env python3

# Script to sample pairs from a given train, dev, and test set of individual papers, and write
# the sampled pairs to disk

import json
from gnlputils import read_dataset
import random
import argparse

# file paths
SHARED_DIR: str = "/projects/instr/19sp/cse481n/GatesNLP/"
TRAIN: str = SHARED_DIR + 'train.txt'
DEV: str = SHARED_DIR + 'dev.txt'
TEST: str = SHARED_DIR + 'test.txt'


def main():
    parser = argparse.ArgumentParser(description='Arguments to be passed into the evaluation script.')
    parser.add_argument('name', type=str, help='the name of the dataset we are creating')

    # total count of each type of pair to be split across train/dev/test
    parser.add_argument('cited', type=int, help='the number of total cited pairs to sample')
    parser.add_argument('one_hop', type=int, help='the number of one-hop pairs to sample')
    parser.add_argument('uncited', type=int, help='the number of total uncited pairs to sample')

    # train/dev/test split
    parser.add_argument('-train_percent', type=float,
                        help='the percentage of the dataset to be used for training', default=0.8)
    parser.add_argument('-dev_percent', type=float,
                        help='the percentage of the dataset to be used for dev', default=0.1)
    parser.add_argument('-test_percent', type=float,
                        help='the percentage of the dataset to be used for test', default=0.1)

    args = parser.parse_args()
    create_pairwise_dataset(args.name, args.cited, args.one_hop, args.uncited, args.train_percent, args.dev_percent,
                            args.test_percent)

def create_pairwise_dataset(name, cited_count, one_hop_count, uncited_count, train_percent, dev_percent, test_percent):
    train_output: str = SHARED_DIR + 'supervised_pairs/' + name + '_pairs_train.txt'
    dev_output: str = SHARED_DIR + 'supervised_pairs/' + name + '_pairs_dev.txt'
    test_output: str = SHARED_DIR + 'supervised_pairs/' + name + '_pairs_test.txt'

    # generate pairs
    train_text, train_citations = read_dataset(TRAIN)
    dev_text, dev_citations = read_dataset(DEV)
    test_text, test_citations = read_dataset(TEST)

    train_cited = generate_cited_pairs(train_text, train_citations)
    dev_cited = generate_cited_pairs_across_datasets(dev_text, train_text, dev_citations)
    test_cited = generate_cited_pairs_across_datasets(test_text, train_text, test_citations)

    train_uncited = generate_uncited_pairs(train_text, train_citations, int(uncited_count * train_percent))
    dev_uncited = generate_uncited_pairs_across_datasets(dev_text, train_text, dev_citations,
                                                         int(uncited_count * dev_percent))
    test_uncited = generate_uncited_pairs_across_datasets(test_text, train_text, test_citations,
                                                          int(uncited_count * test_percent))

    # sample the results down to the desired size
    train_cited = random.sample(train_cited, int(cited_count * train_percent))
    dev_cited = random.sample(dev_cited, int(cited_count * dev_percent))
    test_cited = random.sample(test_cited, int(cited_count * test_percent))

    train_hops = random.sample(generate_one_hops(train_text, train_citations), int(one_hop_count * train_percent))
    dev_hops = random.sample(generate_one_hops_across_datasets(dev_text, train_text, dev_citations, train_citations),
                             int(one_hop_count * dev_percent))
    test_hops = random.sample(generate_one_hops_across_datasets(test_text, train_text, test_citations, train_citations),
                              int(one_hop_count * test_percent))

    # write pairs to disk
    write_output(train_output, train_cited + train_hops + train_uncited)
    write_output(dev_output, dev_cited + dev_hops + dev_uncited)
    write_output(test_output, test_cited + test_hops + test_uncited)


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