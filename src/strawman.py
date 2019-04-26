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
import math

from pprint import pprint

from numpy import dot
from numpy.linalg import norm


PAPERS: str = "dataset_final"


def tf_idf_ranking():
    vectorizer = TfidfVectorizer()
    abstracts = []
    with open(PAPERS, 'rb') as f:
        for line in tqdm(f):
            parsed_json = json.loads(line)
            abstracts.append(parsed_json['paperAbstract'])

    return vectorizer.fit_transform(abstracts).toarray()


def main():
    nlp = spacy.load("en_core_web_sm")
    tokenizer = English().Defaults.create_tokenizer(nlp)

    is_jaccard = len(sys.argv) > 1 and sys.argv[1] == "j"
    lemmatize = len(sys.argv) > 2 and sys.argv[2] == "1"

    lines = []
    with open(PAPERS, 'rb') as f:
        for line in f:
            lines.append(json.loads(line))

    lines.sort(key=lambda x: x['year'])

    ids = extract_keys(lines, 'id')
    abstracts = extract_keys(lines, 'paperAbstract')
    titles = extract_keys(lines, 'title')
    out_citations = extract_keys(lines, 'outCitations')

    train_ids, dev_ids, test_ids = split_data(ids, 0.8, 0.9)
    train_abstracts, dev_abstracts, test_abstracts = split_data(abstracts, 0.8, 0.9)
    train_title, dev_title, test_title = split_data(titles, 0.8, 0.9)
    train_out_citations, dev_out_citations, test_out_citations = split_data(out_citations, 0.8, 0.9)

    print("train size = " + str(len(train_ids)))
    print("dev size = " + str(len(dev_ids)))

    # gets the tokens of the train
    train_token_rows = [set(get_tokens(tokenizer, paper, lemmatize)) for paper in train_abstracts]

    total_count = 0
    citation_counts = dict()
    for i, citations in enumerate(dev_out_citations):
        count = len(set(train_ids) & set(citations))
        total_count += count
        citation_counts[i] = count
    print("total count = " + str(total_count))
    print(dev_title[527])
    pprint(sorted(citation_counts.items(), key = lambda kv:(kv[1], kv[0])))
    print(set(split_data(extract_keys(lines, 'year'), 0.8, 0.9)[0]))

    # get file to write titles too
    f = open("titles_similar_dataset_final.txt", "w", encoding="utf-8")
    f.write("test title, top-10 similar papers\n")

    # evaluation metric
    tfidf_matrix = tf_idf_ranking()
    eval_score = []
    matching_citation_count = 0
    max_rank = float("-inf")
    for i, dev_row in tqdm(enumerate(dev_abstracts)):
        rankings = []
        dev_tokens = set(get_tokens(tokenizer, dev_row, lemmatize))
        # get jaccard similarity for all the papers
        for index, train_tokens in enumerate(train_token_rows):
            a = tfidf_matrix[i + len(train_token_rows)]
            b = tfidf_matrix[index]

            if is_jaccard:
                score = jaccard_similarity(dev_tokens, train_tokens)
            else:
                score = dot(a, b)/(norm(a)*norm(b))
            rankings.append((score, index))
        rankings.sort(key=lambda x: x[0], reverse=True)

        # EVALUATION METRIC LOGIC
        # gets citations if there are any
        out_citations = dev_out_citations[i]
        if len(out_citations):
            # gets the rankings of the training papers in the correct order
            ranking_ids = get_ids(rankings, train_ids)
            list_citations = [out_citation for out_citation in out_citations if out_citation in ranking_ids]

            if len(list_citations):
                matching_citation_count += 1
                rank = ranking_ids.index(list_citations[0]) + 1
                max_rank = max(max_rank, rank)
                eval_score.append(1.0 / rank)
            correct_rankings = list(filter(lambda x: x in train_ids, list_citations))
            print("correct abstracts")
            for j in range(3):
                if j < len(correct_rankings):
                    print(train_abstracts[train_ids.index(correct_rankings[j])])

            # PRINT TOP 10 TITLES PER TEST PAPER
            paper_titles = get_relevant_papers(rankings[:10], train_title)
            f.write(test_title[i] + "\n " + ','.join(list(paper_titles)) + "\n\n")
    print("matching citation count = " + str(matching_citation_count))
    print(eval_score)
    print("max rank = " + str(max_rank))
    print(sum(eval_score) / float(len(test_ids)))
    f.close()


def extract_keys(lines, key: str):
    return [json[key] for json in lines]


def jaccard_similarity(a, b):
    if not a and not b:
        return 0
    c = a & b
    return len(c) / (len(a) + len(b) - len(c))


def get_tokens(tokenizer, paper: str, lemmatize: bool) -> Set[str]:
    if lemmatize:
        tokens = tokenizer(paper.lower())
        return {token.lemma_ for token in tokens if token.is_alpha and not token.is_stop}
    return paper.split()


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
