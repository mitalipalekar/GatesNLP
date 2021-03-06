import json
import argparse

from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

import os, sys, inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from gnlputils import cosine_similarity, extract_keys, split_data, get_from_rankings, get_relevant_titles

import pandas as pd
import numpy as np
import csv
from tqdm import tqdm

from nltk.corpus import stopwords
from nltk import download

download('stopwords')  # Download stopwords list.

GATESNLP_ROOT = '/projects/instr/19sp/cse481n/GatesNLP/'


def vec(words, keys):
    return words.loc[words.index.intersection(keys)].to_numpy().mean(axis=0).transpose()


def glove_embeddings(embeddings_file_name, papers, cosine_similarity_flag, print_titles_flag, is_test):
    # For cosine similarity
    glove_embeddings_file_path = GATESNLP_ROOT + embeddings_file_name
    word2_vec_output_file = glove_embeddings_file_path + '.word2vec'
    glove_embeddings_data = pd.read_csv(glove_embeddings_file_path, sep=" ", index_col=0, header=None,
                                        quoting=csv.QUOTE_NONE)

    # For wmdistance
    if not os.path.exists(word2_vec_output_file):
        glove2word2vec(glove_embeddings_file_path, word2_vec_output_file)
    model = KeyedVectors.load_word2vec_format(word2_vec_output_file, binary=False)
    model.init_sims(replace=True)  # Normalizes the vectors in the word2vec class.

    stop_words = stopwords.words('english')

    lines = []
    dataset_file_path = GATESNLP_ROOT + papers
    with open(dataset_file_path, 'rb') as f:
        for line in tqdm(f, desc='Reading papers'):
            lines.append(json.loads(line))

    lines.sort(key=lambda x: x['year'])

    ids = extract_keys(lines, 'id')
    titles = extract_keys(lines, 'title')
    abstracts = extract_keys(lines, 'paperAbstract')
    out_citations = extract_keys(lines, 'outCitations')

    train_ids, eval_ids = split_data(ids, 0.8, 0.9, is_test)
    train_abstracts, eval_abstracts = split_data(abstracts, 0.8, 0.9, is_test)
    train_titles, eval_titles = split_data(titles, 0.8, 0.9, is_test)
    train_out_citations, eval_out_citations = split_data(out_citations, 0.8, 0.9, is_test)

    # get file to write titles too
    f = open(embeddings_file_name + "error_analysis.txt", "w")
    f.write("test title, top-10 similar papers\n")

    # TODO: changed abstracts to titles
    if cosine_similarity_flag:
        train_titles = [vec(glove_embeddings_data, x.split()) for x in tqdm(train_titles, desc='Train Embeddings')]
        eval_titles = [vec(glove_embeddings_data, x.split()) for x in tqdm(eval_titles, desc='Eval Embeddings')]
        train_titles = list(filter(lambda x: np.isfinite(x).all(), train_titles))
        eval_titles = list(filter(lambda x: np.isfinite(x).all(), eval_titles))

    eval_score = []
    matching_citation_count = 1
    min_rank = float("inf")
    # TODO: changed eval_abstracts -> eval_titles
    for i, eval_abstract in tqdm(list(enumerate(eval_titles)), desc='Generating rankings for evaluation set'):
        rankings = []
        if len(eval_abstract) > 0:
            # TODO: changed train_abstracts -> train_titles
            for j, train_abstract in tqdm(list(enumerate(train_titles)), desc='Iterating through train titles'):
                if cosine_similarity_flag:
                    document_similarity = cosine_similarity(eval_abstract, train_abstract)
                else:
                    train_split = train_abstract.lower().split()
                    eval_split = eval_abstract.lower().split()
                    train_split = [w for w in train_split if w not in stop_words]
                    eval_split = [w for w in eval_split if w not in stop_words]

                    if len(train_split):
                        document_similarity = model.wmdistance(train_split, eval_split)
                rankings.append((document_similarity, j))
            if cosine_similarity_flag:
                rankings.sort(key=lambda x: x[0], reverse=True)
            else:
                rankings.sort(key=lambda x: x[0])

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
                    print("\nEval Score for iteration " + str(i) + ": " + str(1.0 / rank) + "\n")

                # PRINT TOP 10 TITLES PER TEST PAPER
                if print_titles_flag:
                    paper_titles = get_relevant_titles(rankings[:10], train_titles)
                    # print(paper_titles)
                    f.write(
                        str(eval_titles[i]) + ": " + str(1.0 / rank) + "\n " + ','.join(list(paper_titles)) + "\n\n")

    print("matching citation count = " + str(matching_citation_count))
    print(eval_score)
    print("min rank = " + str(min_rank))
    print(sum(eval_score) / matching_citation_count)


def main():
    parser = argparse.ArgumentParser(description='Arguments to be passed into the GloVe embeddings.')
    parser.add_argument('embeddings_file_name', type=str, help='file name of the GloVe vectors')
    parser.add_argument('dataset_file_name', type=str, help='file name of the dataset')
    parser.add_argument('--cosine_similarity_flag', action='store_true',
                        help='whether we want to use cosine similarity')
    parser.add_argument('--print_titles_flag', action='store_true',
                        help='whether to print the top 10 titles')
    parser.add_argument('--test', action='store_true', help='whether to run for test')
    args = parser.parse_args()

    glove_embeddings(args.embeddings_file_name, args.dataset_file_name, args.cosine_similarity_flag,
                     args.print_titles_flag, args.test)


if __name__ == '__main__':
    main()
