#! /usr/bin/env python3

import pickle
from typing import Set

import numpy as np
import os
import pandas as pd
import spacy
from spacy.lang.en import English
from tqdm import tqdm

COUNT = 0
PAPERS: str = "../dataset/papers.csv"
TRAIN_TOKEN_ROWS_PATH = 'train_token_rows.pk'


def main():
    nlp = spacy.load("en_core_web_sm")
    tokenizer = English().Defaults.create_tokenizer(nlp)
    df = pd.read_csv(PAPERS)
    train, dev, test = np.split(df, [int(.8 * len(df)), int(.9 * len(df))])

    if os.path.isfile(TRAIN_TOKEN_ROWS_PATH):
        with open(TRAIN_TOKEN_ROWS_PATH, 'rb') as handle:
            train_token_rows = pickle.load(handle)
    else:
        with open(TRAIN_TOKEN_ROWS_PATH, 'wb') as handle:
            train_token_rows = [set(get_tokens(tokenizer, paper)) for paper in train['paper_text']]
            pickle.dump(train_token_rows, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for dev_row in tqdm(dev['paper_text']):
        rankings = []
        dev_tokens = set(get_tokens(tokenizer, dev_row))
        for index, train_tokens in enumerate(train_token_rows):
            rankings.append((jaccard_similarity(dev_tokens, train_tokens), index))
        rankings.sort(key=lambda x: x[0], reverse=True)
        rankings = rankings[:10]
        print(rankings)


def jaccard_similarity(a, b):
    c = a & b
    return len(c) / (len(a) + len(b) - len(c))


def get_tokens(tokenizer, paper: str) -> Set[str]:
    global COUNT
    COUNT += 1
    print(COUNT)
    paper = paper.lower()
    abstract_index = paper.find('abstract')
    # Remove the headers and citations
    tokens = tokenizer(paper[abstract_index + len('abstract')
                             if abstract_index >= 0 else paper.find('introduction') + len('introduction'):
                             paper.rfind('references')])
    return {token.lemma_ for token in tokens if token.is_alpha and not token.is_stop}


if __name__ == '__main__':
    main()
