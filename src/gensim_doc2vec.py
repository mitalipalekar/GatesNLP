import json

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm, trange

from gnlputils import extract_keys, split_data

PAPERS_PATH = 'extended_large.txt'
WORD_EMBEDDINGS_EVAL = 'doc2vec_eval.pk'
WORD_EMBEDDINGS_TRAIN = 'doc2vec_train.pk'

UNK_THRESHOLD = 3

def generate_word_embeddings():
    lines = []
    with open(PAPERS_PATH, 'rb') as f:
        for line in tqdm(f, desc='Read papers'):
            lines.append(json.loads(line))

    lines.sort(key=lambda x: x['year'])

    abstracts = extract_keys(lines, 'paperAbstract')

    train_abstracts, eval_abstracts = split_data(abstracts, 0.8, 0.9, False)
    dictionary = unk_train(train_abstracts)
    train_docs = create_tagged_doc(train_abstracts, dictionary)

    model = Doc2Vec(dm=1, min_count=1, window=10, size=150, sample=1e-4, negative=10)
    model.build_vocab(train_docs)
    for _ in trange(1):
        model.train(train_docs, epochs=model.iter, total_examples=model.corpus_count)
    print(model.most_similar('embeddings'))

    document_similarity = model.n_similarity(unk_abstract(train_abstracts[0].split(), dictionary), unk_abstract(eval_abstracts[1].split(), dictionary))
    print(document_similarity)


def create_tagged_doc(abstracts: [str], dictionary):
    return [TaggedDocument(unk_abstract(abstract.split(), dictionary), str(i)) for i, abstract in tqdm(enumerate(abstracts), desc='UNKing inputs')]


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

    dictionary = []
    for key, value in word_counts.items():
        if value > UNK_THRESHOLD:
            dictionary.append(key)
    dictionary.append("UNK")
    return dictionary


def main():
    generate_word_embeddings()


if __name__ == '__main__':
    main()
