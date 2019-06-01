#! /usr/bin/env python3

# Script to evaluate the rankings generated by each of our different models. We currently report MRR.

import json
import sys
from typing import Set
import argparse

from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from pathlib import Path

from gnlputils import cosine_similarity, get_from_rankings

# configurations
GPU: int = 1

SHARED_DIR = "/projects/instr/19sp/cse481n/GatesNLP/"
TRAIN: str = SHARED_DIR + "train.txt"
DEV: str = SHARED_DIR + "dev.txt"
TEST: str = SHARED_DIR + "test.txt"
ALLENNLP_MODEL_NAME: str = "relevance_predictor"

BATCH_SIZE: int = 650


def main():
    # read command line arguments
    parser = argparse.ArgumentParser(description='Arguments to be passed into the evaluation script.')
    parser.add_argument('model', type=str, help='the model to evaluate')
    parser.add_argument('--use_titles', action='store_true',
                        help='whether we want to use cosine similarity')
    parser.add_argument('--use_abstracts', action='store_true',
                        help='whether to print the top 10 titles')
    parser.add_argument('--test', action='store_false', help='whether to run for test')
    args = parser.parse_args()

    # evaluate the model as specified
    evaluate_model(args.model, args.use_titles, args.use_abstracts, args.test)


def evaluate_model(model, use_titles, use_abstracts, is_test):
    is_jaccard = model == "j"
    is_tfidf = model == "t"
    is_allennlp = not is_jaccard and not is_tfidf

    # read in train, dev, and test
    train_ids, train_texts, train_out_citations = get_dataset_fields(TRAIN, use_titles, use_abstracts)
    eval_ids, eval_texts, eval_out_citations = get_dataset_fields(TEST if is_test else DEV, use_titles, use_abstracts)

    # gets the tokens of the training set
    train_tokens = [set(paper.split()) for paper in train_texts]

    # calculate statistics on number of papers and citations
    total_count = 0
    citation_counts = dict()
    for i, citations in enumerate(eval_out_citations):
        count = len(set(train_ids) & set(citations))
        total_count += count
        citation_counts[i] = count

    # prepare model-specific objects
    if is_tfidf:
        vectorizer = TfidfVectorizer()
        train_tfidf = vectorizer.fit_transform(train_texts).toarray()
        eval_tfidf = vectorizer.transform(eval_texts).toarray()

    if is_allennlp:
        from allennlp.models.archival import load_archive
        from allennlp.predictors import Predictor
        from gatesnlp.models import pairs_model
        from gatesnlp.dataset_readers import pairs_reader
        from gatesnlp.predictors import predictor

        model_path = SHARED_DIR + "supervised_pairs/" + model + "/model.tar.gz"
        archive = load_archive(model_path, cuda_device=GPU)
        predictor = Predictor.from_archive(archive, ALLENNLP_MODEL_NAME)

    # update output path
    ranking_output_path = SHARED_DIR + "supervised_pairs/rankings_" + model + ".txt"
    output_path = Path(ranking_output_path)
    if output_path.is_file():
        print("Warning: output file will not be written since it already exists")
        ranking_output = None
    else:
        ranking_output = open(ranking_output_path, "w")

    # keep track of each reciprocal rank (and some statistical information)
    eval_score = []
    matching_citation_count = 0
    min_rank = float("inf")

    # create a ranking for each evaluation text and score it.
    for eval_index, eval_text in tqdm(list(enumerate(eval_texts)), desc="Evaluating dev/test set"):
        out_citations = eval_out_citations[eval_index]
        if len(out_citations) > 0:
            # this paper cites some paper in train, so we have a "true" cited paper to rank
            rankings = []

            if is_allennlp:
                # make allennlp predictions in batches before doing the actual ranking loop
                scores = []
                for j in range(0, len(train_texts), BATCH_SIZE):
                    batch = [{"query_paper": eval_text, "candidate_paper": train_text}
                             for train_text in train_texts[j:min(j+BATCH_SIZE, len(train_texts))]]
                    scores.extend(predictor.predict_batch_json(batch))

            # rank all the papers in the training set
            for train_index in range(len(train_ids)):
                # evaluate this train/evaluation pair by whatever model was specified on the command line
                if is_jaccard:
                    score = jaccard_similarity(set(eval_text.split()), train_tokens[train_index])
                elif is_allennlp:
                    class_probs = scores[train_index]['class_probabilities']
                    score = max(class_probs) if scores[train_index]['label'] == "1" else min(class_probs)
                elif is_tfidf:
                    score = cosine_similarity(train_tfidf[train_index], eval_tfidf[eval_index])
                else:
                    raise ValueError("did not provide proper evaluation type as first command line argument")
                rankings.append((score, train_index))
            # sort each ranked paper by its score
            rankings.sort(key=lambda x: x[0], reverse=True)

            # EVALUATION METRIC LOGIC
            # gets citations if there are any
            
            # gets the rankings of the "true" cited pairs in train for this evaluation paper
            ranking_ids = get_from_rankings(rankings, train_ids)
            true_citations = [citation for citation in ranking_ids if citation in out_citations]

            if len(true_citations) > 0:
                # There are citations that we ranked that are actually cited, so add it to the MRR calculation
                # NOTE: with ranking all papers, this will always be true if this paper cites some paper in train
                matching_citation_count += 1
                rank = ranking_ids.index(true_citations[0]) + 1
                min_rank = min(min_rank, rank)
                eval_score.append(1.0 / rank)

                if ranking_output is not None:

                    # log the query paper and the top 10 candidate papers
                    ranking_output.write("QUERY\n" + eval_text + "\n")
                    ranking_output.write("First cited at " + str(rank) + "\n")

                    # log the top correct and incorrect papers
                    ranking_output.write("TOP CITED PAPERS\n")
                    log_top_rankings(true_citations, ranking_ids, train_ids, train_texts, ranking_output)
                    incorrect_rankings = list(filter(lambda x: x not in true_citations, ranking_ids))
                    ranking_output.write("TOP UNCITED PAPERS\n")
                    log_top_rankings(incorrect_rankings, ranking_ids, train_ids, train_texts, ranking_output)
                    ranking_output.write("TOP 20\n")

                    # log the
                    for ranked_index, ranked_tuple in enumerate(rankings[:20]):
                        ranked_score, ranked_train_index = ranked_tuple
                        ranking_output.write("RANK = " + str(ranked_index + 1) + "; score = " + str(ranked_score) +
                                            "; correct = " + str(ranking_ids[ranked_index] in true_citations) +
                                             "\n" + train_texts[ranked_train_index] + "\n")
                    ranking_output.write("\n")

    # calculate final evaluation score for the dataset
    print("matching citation count = " + str(matching_citation_count))
    print(eval_score)
    print("min rank = " + str(min_rank))
    print(sum(eval_score) / matching_citation_count)


# given two sets of words from two papers, calculate the Jaccard similarity.
def jaccard_similarity(a, b):
    if not a and not b:
        return 0
    c = a & b
    return len(c) / (len(a) + len(b) - len(c))


# print top three examples from the given rankings
def log_top_rankings(rankings, ranking_ids, train_ids, train_texts, ranking_output):
    for i in range(3):
        if i < len(rankings):
            paper_index = train_ids.index(rankings[i])
            ranking_output.write("RANK " + str(ranking_ids.index(rankings[i]) + 1) + "\n")
            ranking_output.write(train_texts[paper_index] + "\n")


# read data to return each field as a separate list
def get_dataset_fields(dataset, use_titles, use_abstracts):
    ids = []
    texts = []
    out_citations = []

    with open(dataset, 'rb') as f:
        for line in f:
            parsed_line = json.loads(line)
            if use_abstracts:
                text = parsed_line['paperAbstract']
            elif use_titles:
                text = parsed_line['title']
            else:
                text = parsed_line['title'] + " " + parsed_line['paperAbstract']
            ids.append(parsed_line['id'])
            texts.append(text)
            out_citations.append(parsed_line['outCitations'])
    return ids, texts, out_citations


if __name__ == '__main__':
    main()