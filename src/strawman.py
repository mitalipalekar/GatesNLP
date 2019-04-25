#! /usr/bin/env python3

import sys
import json
from typing import Set

import spacy
from spacy.lang.en import English
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import spatial

from numpy import dot
from numpy.linalg import norm


PAPERS: str = "dataset_final"


def tf_idf_ranking():
    nlp = spacy.load("en_core_web_sm")
    tokenizer = English().Defaults.create_tokenizer(nlp)
    vectorizer = TfidfVectorizer()
    abstracts = []
    count = 0
    with open(PAPERS, 'rb') as f:
        for line in tqdm(f):
            count += 1
            parsed_json = json.loads(line)
            abstracts.append(parsed_json['paperAbstract'])

    return vectorizer.fit_transform(abstracts).toarray()


def main():
    nlp = spacy.load("en_core_web_sm")
    tokenizer = English().Defaults.create_tokenizer(nlp)

    ids = []
    abstracts = []
    titles = []
    out_citations = []
    with open(PAPERS, 'rb') as f:
        for line in tqdm(f):
            ids.append(json.loads(line)['id'])
            abstracts.append(json.loads(line)['paperAbstract'])
            titles.append(json.loads(line)['title'])
            out_citations.append(json.loads(line)['outCitations'])

    train_ids, dev_ids, test_ids = split_data(ids, 0.8, 0.9)
    train_abstracts, dev_abstracts, test_abstracts = split_data(abstracts, 0.8, 0.9)
    train_title, dev_title, test_title = split_data(titles, 0.8, 0.9)
    train_out_citations, dev_out_citations, test_out_citations = split_data(out_citations, 0.8, 0.9)

    # gets the tokens of the train
    train_token_rows = [set(get_tokens(tokenizer, paper)) for paper in train_abstracts]

    # get file to write titles too
    f = open("titles_similar_dataset_final.txt", "w")
    f.write("test title, top-10 similar papers\n")

    # evaluation metric
    tfidf_matrix = tf_idf_ranking()
    eval_score = []
    for i, dev_row in tqdm(enumerate(dev_abstracts)):
        rankings = []
        # dev_tokens = set(get_tokens(tokenizer, dev_row))
        # get jaccard similarity for all the papers
        for index, train_tokens in enumerate(train_token_rows):
            a = tfidf_matrix[i + len(train_token_rows)]
            b = tfidf_matrix[index]
            rankings.append((dot(a, b)/(norm(a)*norm(b)), index))
        rankings.sort(key=lambda x: x[0], reverse=True)

        # EVALUATION METRIC LOGIC
        # gets citations if there are any
        out_citations = dev_out_citations[i]
        if len(out_citations):
            # gets the rankings of the training papers in the correct order
            ranking_ids = get_ids(rankings, train_ids)
            find_citation = next((out_citation for out_citation in out_citations if out_citation in ranking_ids), None)
            if find_citation:
                eval_score.append(1.0 / (ranking_ids.index(find_citation) + 1))

            # PRINT TOP 10 TITLES PER TEST PAPER
            paper_titles = get_relevant_papers(rankings[:10], train_title)
            f.write(test_title[i] + "\n " + ','.join(list(paper_titles)) + "\n\n")
    print(eval_score)
    print(sum(eval_score) / float(len(test_ids)))
    f.close()


def jaccard_similarity(a, b):
    if not a and not b:
        return 0
    c = a & b
    return len(c) / (len(a) + len(b) - len(c))


def get_tokens(tokenizer, paper: str) -> Set[str]:
    tokens = tokenizer(paper.lower())
    return {token.lemma_ for token in tokens if token.is_alpha and not token.is_stop}


def get_relevant_papers(rankings, train_title):
    return [train_title[index] for _, index in rankings]


def get_ids(rankings, train_ids):
    return [train_ids[index] for _, index in rankings]


def split_data(data, start: float, end: float):
    return (data[:int(start * len(data))],
            data[int(start * len(data)): int(end * len(data))],
            data[int(end * len(data)):])


if __name__ == '__main__':
    main()
