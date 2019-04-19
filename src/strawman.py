#! /usr/bin/env python3

import json
from typing import Set

import spacy
from spacy.lang.en import English
import statistics
from tqdm import tqdm

PAPERS: str = "../dataset/papers.json"


def main():
    nlp = spacy.load("en_core_web_sm")
    tokenizer = English().Defaults.create_tokenizer(nlp)

    ids = []
    abstracts = []
    titles = []
    out_citations = []
    with open(PAPERS, 'r') as f:
        for line in tqdm(f):
            ids.append(json.loads(line)['id'])
            abstracts.append(json.loads(line)['paperAbstract'])
            titles.append(json.loads(line)['title'])
            out_citations.append(json.loads(line)['outCitations'])

    train_ids, dev_ids, test_ids = (ids[:int(0.8 * len(ids))],
                        ids[int(0.8 * len(ids)): int(0.9 * len(ids))],
                        ids[int(0.9 * len(ids)):])
    train_abstracts, dev_abstracts, test_abstracts = (abstracts[:int(0.8 * len(abstracts))],
                        abstracts[int(0.8 * len(abstracts)): int(0.9 * len(abstracts))],
                        abstracts[int(0.9 * len(abstracts)):])
    train_title, dev_title, test_title = (titles[:int(0.8 * len(titles))],
                                          titles[int(0.8 * len(titles)): int(0.9 * len(titles))],
                                          titles[int(0.9 * len(titles)):])
    train_out_citations, dev_out_citations, test_out_citations = (out_citations[:int(0.8 * len(out_citations))],
                                          out_citations[int(0.8 * len(out_citations)): int(0.9 * len(out_citations))],
                                          out_citations[int(0.9 * len(out_citations)):])

    # gets the tokens of the train
    train_token_rows = [set(get_tokens(tokenizer, paper)) for paper in train_abstracts]

    # get file to write titles too
    f = open("titles_similar.txt", "w")
    f.write("test title, top-10 similar papers\n")

    # evaluation metric
    eval_score = []
    for i, test_row in tqdm(enumerate(test_abstracts)):
        rankings = []
        test_tokens = set(get_tokens(tokenizer, test_row))
        # get jaccard similarity for all the papers
        for index, train_tokens in enumerate(train_token_rows):
            rankings.append((jaccard_similarity(test_tokens, train_tokens), index))
        rankings.sort(key=lambda x: x[0], reverse=True)


        # EVALUATION METRIC LOGIC
        # gets citations if there are any
        out_citations = test_out_citations[i]
        if len(out_citations) == 0:
            continue
        # gets the rankings of the training papers in the correct order
        ranking_ids = get_ids(rankings, train_ids)
        for out_citation in out_citations:
            if out_citation in ranking_ids:
                eval_score.append(1.0/ (ranking_ids.index(out_citation) + 1))

        # PRINT TOP 10 TITLES PER TEST PAPER
        paper_titles = get_relevant_papers(rankings[:10], train_title)
        f.write(test_title[i] + "\n " + ','.join(list(paper_titles))+"\n\n")
        # print(test_title[i])
        # print(list(paper_titles))
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
    titles = [];
    for rank, index in rankings:
        titles.append(train_title[index])
    return titles


def get_ids(rankings, train_ids):
    ids = []
    for rankings, index in rankings:
        ids.append(train_ids[index])
    return ids


if __name__ == '__main__':
    main()
