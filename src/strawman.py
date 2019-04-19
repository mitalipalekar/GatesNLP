#! /usr/bin/env python3

import json
from typing import Set

import spacy
from spacy.lang.en import English
from tqdm import tqdm

PAPERS: str = "../dataset/papers.json"


def main():
    nlp = spacy.load("en_core_web_sm")
    tokenizer = English().Defaults.create_tokenizer(nlp)
    abstracts = []
    with open(PAPERS, 'r') as f:
        for line in tqdm(f):
            abstracts.append(json.loads(line)['paperAbstract'])

    train, dev, test = (abstracts[:int(0.8 * len(abstracts))],
                        abstracts[int(0.8 * len(abstracts)): int(0.9 * len(abstracts))],
                        abstracts[int(0.9 * len(abstracts)):])

    train_token_rows = [set(get_tokens(tokenizer, paper)) for paper in train]
    similarity_sum = 0
    for dev_row in tqdm(test):
        rankings = []
        dev_tokens = set(get_tokens(tokenizer, dev_row))
        for index, train_tokens in enumerate(train_token_rows):
            rankings.append((jaccard_similarity(dev_tokens, train_tokens), index))
        rankings.sort(key=lambda x: x[0], reverse=True)
        similarity_sum += sum(r[0] for r in rankings[:10])
    print(similarity_sum)


def jaccard_similarity(a, b):
    if not a and not b:
        return 0
    c = a & b
    return len(c) / (len(a) + len(b) - len(c))


def get_tokens(tokenizer, paper: str) -> Set[str]:
    tokens = tokenizer(paper.lower())
    return {token.lemma_ for token in tokens if token.is_alpha and not token.is_stop}


if __name__ == '__main__':
    main()
