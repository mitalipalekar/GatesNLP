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

    is_test = len(sys.argv) > 1 and sys.argv[1] == "test"
    is_jaccard = len(sys.argv) > 2 and sys.argv[2] == "j"
    lemmatize = len(sys.argv) > 3 and sys.argv[3] == "1"

    lines = []
    with open(PAPERS, 'rb') as f:
        for line in f:
            lines.append(json.loads(line))

    lines.sort(key=lambda x: x['year'])

    ids = extract_keys(lines, 'id')
    abstracts = extract_keys(lines, 'paperAbstract')
    titles = extract_keys(lines, 'title')
    out_citations = extract_keys(lines, 'outCitations')

    train_ids, eval_ids = split_data(ids, 0.8, 0.9, is_test)
    train_abstracts, eval_abstracts = split_data(abstracts, 0.8, 0.9, is_test)
    train_title, eval_title = split_data(titles, 0.8, 0.9, is_test)
    train_out_citations, eval_out_citations = split_data(out_citations, 0.8, 0.9, is_test)

    # gets the tokens of the training set
    train_token_rows = [set(get_tokens(tokenizer, paper, lemmatize)) for paper in train_abstracts]

    total_count = 0
    citation_counts = dict()
    for i, citations in enumerate(eval_out_citations):
        count = len(set(train_ids) & set(citations))
        total_count += count
        citation_counts[i] = count

    print("train size = " + str(len(train_ids)))
    print("dev size = " + str(len(eval_ids)))

    print("total count = " + str(total_count))
    print(eval_title[527])
    pprint(sorted(citation_counts.items(), key = lambda kv:(kv[1], kv[0])))
    print(set(split_data(extract_keys(lines, 'year'), 0.8, 0.9, is_test)[0]))

    # get file to write titles too
    f = open("titles_similar_dataset_final.txt", "w", encoding="utf-8")
    f.write("test title, top-10 similar papers\n")

    # evaluation metric
    tfidf_matrix = tf_idf_ranking()
    eval_score = []
    matching_citation_count = 0
    max_rank = float("-inf")
    for i, eval_row in tqdm(enumerate(eval_abstracts)):
        rankings = []
        dev_tokens = set(get_tokens(tokenizer, eval_row, lemmatize))
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
        out_citations = eval_out_citations[i]
        if len(out_citations):
            # gets the rankings of the training papers in the correct order
            ranking_ids = get_ids(rankings, train_ids)
            list_citations = [out_citation for out_citation in out_citations if out_citation in ranking_ids]

            if len(list_citations):
                matching_citation_count += 1
                rank = ranking_ids.index(list_citations[0]) + 1
                max_rank = max(max_rank, rank)
                eval_score.append(1.0 / rank)

            print("PAPER " + str(i))
            print(eval_title[i])
            print(eval_abstracts[i])

            correct_rankings = list(filter(lambda x: x in train_ids, list_citations))
            print("correct papers")
            print_top_three(correct_rankings, train_ids, train_title, train_abstracts)

            incorrect_rankings = list(filter(lambda x: x not in correct_rankings, ranking_ids))
            print("incorrect papers")
            print_top_three(incorrect_rankings, train_ids, train_title, train_abstracts)
            print()

            # PRINT TOP 10 TITLES PER TEST PAPER
            # paper_titles = get_relevant_papers(rankings[:10], train_title)
            # f.write(eval_title[i] + "\n " + ','.join(list(paper_titles)) + "\n\n")
    print("matching citation count = " + str(matching_citation_count))
    print(eval_score)
    print("max rank = " + str(max_rank))
    print(sum(eval_score) / float(len(eval_ids)))
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


def split_data(data, dev_start: float, test_start: float, is_test: bool):
    return (data[:int(dev_start * len(data))],
            data[int(test_start * len(data)):] if is_test
            else data[int(dev_start * len(data)): int(test_start * len(data))])

def print_top_three(rankings, train_ids, train_title, train_abstracts):
    for i in range(3):
        if i < len(rankings):
            paper_index = train_ids.index(rankings[i])
            print(train_title[paper_index])
            print(train_abstracts[paper_index])


if __name__ == '__main__':
    main()
