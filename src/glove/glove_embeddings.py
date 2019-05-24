import json
import argparse

from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

from src.gnlputils import cosine_similarity, extract_keys, split_data, get_from_rankings
import pandas as pd
import numpy as np
import csv
from tqdm import tqdm

GLOVE_INPUT_FILE_PATH = '/projects/instr/19sp/cse481n/GatesNLP/'
WORD2VEC_OUTPUT_FILE = 'glove.6B.50d.txt.word2vec'


def vec(words, keys):
    return words.loc[words.index.intersection(keys)].to_numpy().mean(axis=0).transpose()


def glove_embeddings(embeddings_file_name, papers, cosine_similarity_flag):
    # For cosine similarity
    glove_embeddings_file_path = GLOVE_INPUT_FILE_PATH + embeddings_file_name;
    glove_embeddings = pd.read_csv(glove_embeddings_file_path, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

    # For wmdistance
    glove2word2vec(glove_embeddings_file_path, WORD2VEC_OUTPUT_FILE)
    model = KeyedVectors.load_word2vec_format(WORD2VEC_OUTPUT_FILE, binary=False)

    lines = []
    dataset_file_path = GLOVE_INPUT_FILE_PATH + papers
    with open(dataset_file_path, 'rb') as f:
        for line in tqdm(f, desc='Reading papers'):
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

    if cosine_similarity_flag:
        eval_abstracts = [vec(glove_embeddings, x.split()) for x in tqdm(eval_abstracts, desc='Eval Embeddings')]
        train_abstracts = [vec(glove_embeddings, x.split()) for x in tqdm(train_abstracts, desc='Train Embeddings')]
        eval_abstracts = filter(lambda x: np.isfinite(x).all(), eval_abstracts)
        train_abstracts = filter(lambda x: np.isfinite(x).all(), train_abstracts)

    eval_score = []
    matching_citation_count = 1
    min_rank = float("inf")
    for i, eval_abstract in tqdm(list(enumerate(eval_abstracts[:3])), desc='Generating rankings for evaluation set'):
        rankings = []
        eval_split = eval_abstract.lower().split()
        if len(eval_split):
            for j, train_abstract in tqdm(list(enumerate(train_abstracts)), desc='Iterating through train titles'):
                if cosine_similarity_flag:
                    document_similarity = cosine_similarity(eval_abstract, train_abstract)
                else:
                    train_split = train_abstract.lower().split()
                    if len(train_split):
                        document_similarity = model.wmdistance(train_split, eval_split)
                        rankings.append((document_similarity, j))
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
    parser = argparse.ArgumentParser(description='Arguments to be passed into the GloVe embeddings.')
    parser.add_argument('embeddings_file_name', type=str, help = 'file name of the GloVe vectors')
    parser.add_argument('dataset_file_name', type=str, help='file name of the dataset')
    parser.add_argument('--cosine_similarity_flag', action='store_true', help = 'whether we want to use cosine similiarty')
    args = parser.parse_args()

    glove_embeddings(args.embeddings_file_name, args.dataset_file_name, args.cosine_similarity_flag)


if __name__ == '__main__':
    main()
