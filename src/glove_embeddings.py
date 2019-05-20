import json

from collections import Counter
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gnlputils import extract_keys, split_data, get_from_rankings
from tqdm import tqdm

GLOVE_INPUT_FILE_PATH = '/projects/instr/19sp/cse481n/GatesNLP/glove.6B.50d.txt'
WORD2VEC_OUTPUT_FILE = 'glove.6B.50d.txt.word2vec'

UNK_THRESHOLD = 3


def unk_abstract(abstract, dictionary):
    return [word if word in dictionary else 'UNK' for word in abstract]


def unk_train(train_abstracts):
    word_counts = {}
    for abstract in train_abstracts:
        for word in abstract.split():
            if word in word_counts:
                word_counts[word] = word_counts.get(word) + 1
            else:
                word_counts[word] = 1

    return generate_dictionary(word_counts)


def generate_dictionary(word_counts):
    dictionary = {'UNK'}
    for key, value in word_counts.items():
        if value > UNK_THRESHOLD:
            dictionary.add(key)
    return dictionary


def glove_embeddings(papers):
    glove2word2vec(GLOVE_INPUT_FILE_PATH, WORD2VEC_OUTPUT_FILE)
    model = KeyedVectors.load_word2vec_format(WORD2VEC_OUTPUT_FILE, binary=False)

    lines = []
    with open(papers, 'rb') as f:
        for line in tqdm(f, desc='Read papers'):
            lines.append(json.loads(line))

    lines.sort(key=lambda x: x['year'])

    ids = extract_keys(lines, 'id')
    abstracts = extract_keys(lines, 'paperAbstract')
    out_citations = extract_keys(lines, 'outCitations')

    # TODO: DO NOT HARDCODE THIS
    is_test = False

    train_ids, eval_ids = split_data(ids, 0.8, 0.9, is_test)
    train_abstracts, eval_abstracts = split_data(abstracts, 0.8, 0.9, is_test)
    train_out_citations, eval_out_citations = split_data(out_citations, 0.8, 0.9, is_test)

    dictionary = unk_train(train_abstracts)

    # NOTE: Make sure to always UNK everything!
    eval_score = []
    matching_citation_count = 1
    min_rank = float("inf")
    for i, eval_abstract in tqdm(list(enumerate(eval_abstracts)), desc='generating rankings for evaluation set'):
        rankings = []
        for j, train_abstract in tqdm(list(enumerate(train_abstracts)), desc='iterating through train abstracts'):
            eval_split = eval_abstract.lower().split()
            train_split = train_abstract.lower().split()
            if len(eval_split) and len(train_split):
                document_similarity = model.wmdistance(unk_abstract(train_split, dictionary),
                                                       unk_abstract(eval_split, dictionary))
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
    glove_embeddings('/projects/instr/19sp/cse481n/GatesNLP/extended_dataset.txt')


if __name__ == '__main__':
    main()
