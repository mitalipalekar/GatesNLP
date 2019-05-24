#! /usr/bin/env python3
import json
from tqdm import tqdm
from gnlputils import read_dataset
import random

TOTAL_PAIRS = 120000
TRAIN_PERCENT = 0.8
DEV_PERCENT = 0.1
TEST_PERCENT = 0.1

SHARED_DIR: str = "/projects/instr/19sp/cse481n/GatesNLP/"
TRAIN: str = SHARED_DIR + 'train.txt'
DEV: str = SHARED_DIR + 'dev.txt'
TEST: str = SHARED_DIR + 'test.txt'
TRAIN_OUTPUT: str = SHARED_DIR + 'supervised_pairs/large_pairs_train.txt'
DEV_OUTPUT: str = SHARED_DIR + 'supervised_pairs/large_pairs_dev.txt'
TEST_OUTPUT: str = SHARED_DIR + 'supervised_pairs/large_pairs_test.txt'


def main():
    train_text, train_citations = read_dataset(TRAIN)
    dev_text, dev_citations = read_dataset(DEV)
    test_text, test_citations = read_dataset(TEST)

    train_cited = generate_cited_pairs(train_text, train_citations)
    dev_cited = generate_cited_pairs_across_datasets(dev_text, train_text, dev_citations)
    test_cited = generate_cited_pairs_across_datasets(test_text, train_text, test_citations)

    # take minimum count of cited papers as the number of cited/uncited pairs we are retrieving from the dataset
    train_cited = random.sample(train_cited, int(TOTAL_PAIRS * TRAIN_PERCENT / 2))
    dev_cited = random.sample(dev_cited, int(TOTAL_PAIRS * DEV_PERCENT / 2))
    test_cited = random.sample(test_cited, int(TOTAL_PAIRS * TEST_PERCENT / 2))

    train_uncited = generate_uncited_pairs(train_text, train_citations, int(TOTAL_PAIRS * TRAIN_PERCENT / 2))
    dev_uncited = generate_uncited_pairs_across_datasets(dev_text, train_text, dev_citations,
                                                         int(TOTAL_PAIRS * DEV_PERCENT / 2))
    test_uncited = generate_uncited_pairs_across_datasets(test_text, train_text, test_citations,
                                                          int(TOTAL_PAIRS * TEST_PERCENT / 2))

    write_output(TRAIN_OUTPUT, train_cited + train_uncited)
    write_output(DEV_OUTPUT, dev_cited + dev_uncited)
    write_output(TEST_OUTPUT, test_cited + test_uncited)





def generate_cited_pairs(text, out_citations):
    return generate_cited_pairs_across_datasets(text, text, out_citations)


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


def generate_uncited_pairs(text, out_citations, pair_count):
    return generate_uncited_pairs_across_datasets(text, text, out_citations, pair_count)


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


def write_output(filename, pairs):
    out = open(filename, 'w')
    random.shuffle(pairs)
    for pair in pairs:
        out.write(json.dumps(pair) + '\n')


if __name__ == '__main__':
    main()
