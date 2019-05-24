import json
import sys

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm, trange

from gnlputils import extract_keys, split_data, get_from_rankings

WORD_EMBEDDINGS_EVAL = 'doc2vec_eval.pk'
WORD_EMBEDDINGS_TRAIN = 'doc2vec_train.pk'

UNK_THRESHOLD = 3


def generate_word_embeddings(papers):
    global document_similarity
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

    dictionary = unk_train(train_abstracts)
    train_docs = create_tagged_doc(train_abstracts, dictionary)

    model = Doc2Vec(workers=11, min_count=5, window=10, size=100, alpha=0.025, iter=20)
    model.build_vocab(train_docs)
    model.train(train_docs, epochs=model.iter, total_examples=model.corpus_count)

    # NOTE: Make sure to always UNK everything!
    eval_score = []
    matching_citation_count = 1
    min_rank = float("inf")

    for i, eval_abstract in tqdm(list(enumerate(eval_abstracts[:2])), desc='generating rankings for evaluation set'):
        eval_split = eval_abstract.lower().split()

        if len(eval_split):
            eval_doc_vec = model.infer_vector(eval_split, steps=50, alpha=0.25)
            rankings = model.docvecs.most_similar(positive=[eval_doc_vec])
            rankings = [(score, int(index)) for index, score in rankings]

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


def create_tagged_doc(abstracts: [str], dictionary):
    return [TaggedDocument(unk_abstract(abstract.split(), dictionary), str(i)) for i, abstract in tqdm(enumerate(abstracts), desc='UNKing inputs')]


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


def main():
    generate_word_embeddings(sys.argv[1])


if __name__ == '__main__':
    main()
