import json
import sys

from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gnlputils import extract_keys, split_data, get_from_rankings
from tqdm import tqdm

GLOVE_INPUT_FILE_PATH = '/projects/instr/19sp/cse481n/GatesNLP/glove.6B.50d.txt'
WORD2VEC_OUTPUT_FILE = 'glove.6B.50d.txt.word2vec'


def glove_embeddings(papers):
    glove2word2vec(GLOVE_INPUT_FILE_PATH, WORD2VEC_OUTPUT_FILE)
    model = KeyedVectors.load_word2vec_format(WORD2VEC_OUTPUT_FILE, binary=False)

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

    # NOTE: Make sure to always UNK everything!
    eval_score = []
    matching_citation_count = 1
    min_rank = float("inf")
    for i, eval_abstract in tqdm(list(enumerate(eval_titles)), desc='generating rankings for evaluation set'):
        rankings = []
        eval_split = eval_abstract.lower().split()
        if len(eval_split):
            for j, train_abstract in tqdm(list(enumerate(train_titles)), desc='iterating through train titlesg'):
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
                print("rank for iteration " + str(i) + ": " + str(rank))
                print("eval score for iteration " + str(i) + ": " + str(eval_score))

    print("matching citation count = " + str(matching_citation_count))
    print(eval_score)
    print("min rank = " + str(min_rank))
    print(sum(eval_score) / matching_citation_count)


def main():
    glove_embeddings(sys.argv[1])


if __name__ == '__main__':
    main()
