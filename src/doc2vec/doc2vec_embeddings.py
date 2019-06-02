import json
import sys

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm, trange

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from gnlputils import extract_keys, split_data, get_from_rankings

DATASET_INPUT_FILE_PATH = '/projects/instr/19sp/cse481n/GatesNLP/'
WORD_EMBEDDINGS_EVAL = 'doc2vec_eval.pk'
WORD_EMBEDDINGS_TRAIN = 'doc2vec_train.pk'

# UNK_THRESHOLD = 3


def generate_word_embeddings(papers):
    global document_similarity
    lines = []
    with open(DATASET_INPUT_FILE_PATH + papers, 'rb') as f:
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

    # dictionary = unk_train(train_abstracts)
    train_docs = create_tagged_doc(train_abstracts)

    model = Doc2Vec(workers=11, min_count=5, window=10, size=100, alpha=0.025, iter=20)
    model.build_vocab(train_docs)
    model.train(train_docs, epochs=model.iter, total_examples=model.corpus_count)

    eval_score = []
    matching_citation_count = 1
    min_rank = float("inf")

    # TODO: changed eval_abstracts -> eval_titles
    for i, eval_abstract in tqdm(list(enumerate(eval_titles[:10])), desc='Generating rankings for evaluation set'):
        rankings = []
        eval_split = eval_abstract.lower().split()

        if len(eval_split):
            # TODO: changed train_abstracts -> train_titles
            for j, train_abstract in tqdm(list(enumerate(train_titles)), desc='Iterating through train titles'):
                train_split = train_abstract.lower().split()
                if len(train_split):
                    document_similarity = model.wmdistance(train_split, eval_split)
                    rankings.append((document_similarity, j))
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

    print("matching citation count = " + str(matching_citation_count))
    print(eval_score)
    print("min rank = " + str(min_rank))
    print(sum(eval_score) / matching_citation_count)


def create_tagged_doc(abstracts: [str]):
    return [TaggedDocument(abstract.lower().split(), str(i)) for i, abstract in tqdm(enumerate(abstracts), desc='UNKing inputs')]


def main():
    generate_word_embeddings(sys.argv[1])


if __name__ == '__main__':
    main()
