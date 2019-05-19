import json
import sys

from tqdm import tqdm

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

from gnlputils import extract_keys, split_data, get_from_rankings

GLOVE_INPUT_FILE_PATH = '../dataset/glove/glove.6B.50d.txt'
WORD2VEC_OUTPUT_FILE = 'glove.6B.50d.txt.word2vec'

UNK_THRESHOLD = 3


def unk_abstract(abstract, dictionary):
    unked_abstract = []
    for word in abstract:
        if word in dictionary:
            unked_abstract.append(word)
        else:
            unked_abstract.append("UNK")
    return unked_abstract


def unk_train(train_abstracts):
    word_counts = {}
    for abstract in train_abstracts:
        for word in abstract.split():
            if word in word_counts:
                word_counts[word] = word_counts.get(word) + 1;
            else:
                word_counts[word] = 1;

    return generate_dictionary(word_counts)


def generate_dictionary(word_counts):
    dictionary = []
    for key, value in word_counts.items():
        if value > UNK_THRESHOLD:
            dictionary.append(key)
    dictionary.append("UNK")
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
    for i, eval_abstract in tqdm(enumerate(eval_abstracts[:1]), desc='generating rankings for evaluation set'):
        rankings = []
        for j, train_abstract in tqdm(enumerate(train_abstracts), desc='iterating through train abstracts'):
            if len(eval_abstract.split()) and len(train_abstract.split()):
                document_similarity = KeyedVectors.similarity_unseen_docs(model, unk_abstract(train_abstract.split(), dictionary),
                                                         unk_abstract(eval_abstract.split(), dictionary))
            rankings.append((document_similarity, j))
        rankings.sort(key=lambda x: x[0], reverse=True)

        out_citations = eval_out_citations[i]
        if len(out_citations):
            # gets the rankings of the training papers in the correct order
            ranking_ids = get_from_rankings(rankings, train_ids)
            true_citations = [citation for citation in ranking_ids if citation in out_citations]
            print(true_citations)
            if len(true_citations):
                matching_citation_count += 1
                rank = ranking_ids.index(true_citations[0]) + 1
                min_rank = min(min_rank, rank)
                eval_score.append(1.0 / rank)
        break

    print("matching citation count = " + str(matching_citation_count))
    print(eval_score)
    print("min rank = " + str(min_rank))
    print(sum(eval_score) / matching_citation_count)


def main():
    glove_embeddings(sys.argv[1])


if __name__ == '__main__':
    main()