import json
import sys

from gnlputils import cosine_similarity, extract_keys, split_data, get_from_rankings
import pandas as pd
import numpy as np
import csv
from tqdm import tqdm

GLOVE_INPUT_FILE_PATH = '/projects/instr/19sp/cse481n/GatesNLP/glove.6B.50d.txt'


def vec(words, keys):
    return words.loc[words.index.intersection(keys)].to_numpy().mean(axis=0).transpose()


def glove_embeddings(papers):
    words = pd.read_csv(GLOVE_INPUT_FILE_PATH, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
    lines = []
    with open(papers, 'rb') as f:
        for line in tqdm(f, desc='Read papers'):
            lines.append(json.loads(line))

    lines.sort(key=lambda x: x['year'])

    ids = extract_keys(lines, 'id')
    titles = extract_keys(lines, 'title')
    abstracts = extract_keys(lines, 'paperAbstract')
    out_citations = extract_keys(lines, 'outCitations')

    # TODO: DO NOT HARDCODE THIS
    is_test = False

    train_ids, eval_ids = split_data(ids, 0.8, 0.9, is_test)
    train_abstracts, eval_abstracts = split_data(abstracts, 0.8, 0.9, is_test)
    train_titles, eval_titles = split_data(titles, 0.8, 0.9, is_test)
    train_out_citations, eval_out_citations = split_data(out_citations, 0.8, 0.9, is_test)

    eval_score = []
    matching_citation_count = 1
    min_rank = float("inf")
    eval_abstracts = [vec(words, x.split()) for x in tqdm(eval_abstracts, desc='Eval Embeddings')]
    train_abstracts = [vec(words, x.split()) for x in tqdm(train_abstracts, desc='Train Embeddings')]
    eval_abstracts = filter(lambda x: np.isfinite(x).all(), eval_abstracts)
    train_abstracts = filter(lambda x: np.isfinite(x).all(), train_abstracts)
    for i, eval_abstract in tqdm(list(enumerate(eval_abstracts)), desc='Generating rankings for evaluation set'):
        rankings = []
        for j, train_abstract in tqdm(list(enumerate(train_abstracts)), desc='Iterating through train titles'):
            document_similarity = cosine_similarity(eval_abstract, train_abstract)
            rankings.append((document_similarity, j))
        rankings.sort(key=lambda x: x[0], reverse=True)

        out_citations = eval_out_citations[i]
        if len(out_citations):
            # gets the rankings of the training papers in the correct order
            ranking_ids = get_from_rankings(rankings, train_ids)
            true_citations = [citation for citation in ranking_ids if citation in out_citations]
            if len(true_citations):
                matching_citation_count += 1
                rank = ranking_ids.index(true_citations[0]) + 1
                min_rank = min(min_rank, rank)
                eval_score.append(1.0 / rank)

    print("matching citation count = " + str(matching_citation_count))
    print(eval_score)
    print("min rank = " + str(min_rank))
    print(sum(eval_score) / matching_citation_count)


def main():
    glove_embeddings(sys.argv[1])


if __name__ == '__main__':
    main()
