#! /usr/bin/env python3

import json
import sys
from typing import Set

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en import English
from tqdm import tqdm

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from gatesnlp.models import pairs_model
from gatesnlp.dataset_readers import pairs_reader
from gatesnlp.predictors import predictor

from gnlputils import cosine_similarity, extract_keys, get_from_rankings, split_data

GPU: int = -1
PAPERS: str = "/projects/instr/19sp/cse481n/GatesNLP/extended_dataset.txt"
MODEL: str = "/projects/instr/19sp/cse481n/GatesNLP/supervised_pairs/quadruple/model.tar.gz"
MODEL_NAME: str = "relevance_predictor"
BATCH_SIZE: int = 650


def create_tf_idf_matrix(text):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(text).toarray()


def main():
    nlp = spacy.load("en_core_web_sm")
    tokenizer = English().Defaults.create_tokenizer(nlp)

    is_jaccard = len(sys.argv) > 1 and sys.argv[1] == "j"
    is_allennlp = len(sys.argv) > 1 and sys.argv[1] == "a"
    is_tfidf = len(sys.argv) > 1 and sys.argv[1] == "t"

    use_titles = len(sys.argv) > 2 and sys.argv[2] == "title"
    use_abstracts = len(sys.argv) > 2 and sys.argv[2] == "abstract"

    is_test = len(sys.argv) > 3 and sys.argv[3] == "test"
    lemmatize = len(sys.argv) > 4 and sys.argv[4] == "1"

    if is_allennlp:
        archive = load_archive(MODEL, cuda_device=GPU)
        predictor = Predictor.from_archive(archive, MODEL_NAME)

    lines = []
    with open(PAPERS, 'rb') as f:
        for line in f:
            lines.append(json.loads(line))

    lines.sort(key=lambda x: x['year'])

    ids = extract_keys(lines, 'id')
    if use_abstracts:
        text = extract_keys(lines, 'paperAbstract')
    elif use_titles:
        text = extract_keys(lines, 'title')
    else:
        text = [paper[0] + " " + paper[1] for paper
                in zip(extract_keys(lines, 'title'), extract_keys(lines, 'paperAbstract'))]
    out_citations = extract_keys(lines, 'outCitations')

    train_ids, eval_ids = split_data(ids, 0.8, 0.9, is_test)
    train_texts, eval_texts = split_data(text, 0.8, 0.9, is_test)
    train_out_citations, eval_out_citations = split_data(out_citations, 0.8, 0.9, is_test)

    # gets the tokens of the training set
    train_token_rows = [set(get_tokens(tokenizer, paper, lemmatize)) for paper in train_texts]

    total_count = 0
    citation_counts = dict()
    for i, citations in enumerate(eval_out_citations):
        count = len(set(train_ids) & set(citations))
        total_count += count
        citation_counts[i] = count

    # get file to write titles too
    f = open("titles_similar_dataset_final.txt", "w", encoding="utf-8")
    f.write("test title, top-10 similar papers\n")

    # calculate our evaluation metric
    if is_tfidf:
        tfidf_matrix = create_tf_idf_matrix(text)
    eval_score = []
    matching_citation_count = 0
    min_rank = float("inf")
    for i, eval_text in tqdm(list(enumerate(eval_texts)), desc="Evaluating dev/test set"):
        out_citations = eval_out_citations[i]
        if len(out_citations) > 0:
            rankings = []

            # rank all the papers in the training set
            if is_allennlp:
                scores = []
                for j in range(0, len(train_texts), BATCH_SIZE):
                    batch = [{"query_paper": eval_text, "candidate_paper": train_text}
                             for train_text in train_texts[j:min(j+BATCH_SIZE, len(train_texts))]]
                    scores.extend(predictor.predict_batch_json(batch))
            for train_index, train_tokens in enumerate(train_token_rows):
                if is_jaccard:
                    eval_tokens = set(get_tokens(tokenizer, eval_text, lemmatize))
                    score = jaccard_similarity(eval_tokens, train_tokens)
                elif is_allennlp:
                    score = scores[train_index]['class_probabilities'][1]
                elif is_tfidf:
                    eval_index = i + len(train_token_rows)
                    if is_test:
                        eval_index += len(ids) - int(0.9 * len(ids))
                    a = tfidf_matrix[eval_index]
                    b = tfidf_matrix[train_index]
                    score = cosine_similarity(a, b)
                else:
                    raise ValueError("did not provide proper evaluation type as first command line argument")
                rankings.append((score, train_index))
            rankings.sort(key=lambda x: x[0], reverse=True)

            # EVALUATION METRIC LOGIC
            # gets citations if there are any
            
            # gets the rankings of the training papers in the correct order
            ranking_ids = get_from_rankings(rankings, train_ids)
            true_citations = [citation for citation in ranking_ids if citation in out_citations]

            if len(true_citations):
                matching_citation_count += 1
                rank = ranking_ids.index(true_citations[0]) + 1
                min_rank = min(min_rank, rank)
                eval_score.append(1.0 / rank)

                """
                print("PAPER " + str(i))
                print(eval_title[i])
                print(eval_abstracts[i])
                print("correct papers")
                print_top_three(true_citations, ranking_ids, train_ids, train_title, train_abstracts)
                incorrect_rankings = list(filter(lambda x: x not in true_citations, ranking_ids))
                print("incorrect papers")
                print_top_three(incorrect_rankings, ranking_ids, train_ids, train_title, train_abstracts)
                print()
                """

                # PRINT TOP 10 TITLES PER TEST PAPER
                # paper_titles = get_relevant_papers(rankings[:10], train_title)
                # f.write(eval_title[i] + "\n " + ','.join(list(paper_titles)) + "\n\n")
    print("matching citation count = " + str(matching_citation_count))
    print(eval_score)
    print("min rank = " + str(min_rank))
    print(sum(eval_score) / matching_citation_count)
    f.close()


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


def print_top_three(rankings, ranking_ids, train_ids, train_title, train_abstracts):
    for i in range(3):
        if i < len(rankings):
            paper_index = train_ids.index(rankings[i])
            print(ranking_ids.index(rankings[i]) + 1)
            print(train_title[paper_index])
            print(train_abstracts[paper_index])


if __name__ == '__main__':
    main()