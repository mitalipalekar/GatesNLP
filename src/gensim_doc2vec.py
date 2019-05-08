import json

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm, trange

from src.gnlputils import extract_keys, split_data

PAPERS_PATH = 'extended_dataset.txt'
WORD_EMBEDDINGS_EVAL = 'doc2vec_eval.pk'
WORD_EMBEDDINGS_TRAIN = 'doc2vec_train.pk'


def generate_word_embeddings():
    lines = []
    with open(PAPERS_PATH, 'rb') as f:
        for line in tqdm(f, desc='Read papers'):
            lines.append(json.loads(line))

    lines.sort(key=lambda x: x['year'])

    abstracts = extract_keys(lines, 'paperAbstract')

    train_abstracts, eval_abstracts = split_data(abstracts, 0.8, 0.9, False)
    train_docs = create_tagged_doc(train_abstracts)

    model = Doc2Vec(dm=1, min_count=1, window=10, size=150, sample=1e-4, negative=10)
    model.build_vocab(train_docs)
    for _ in trange(1):
        model.train(train_docs, epochs=model.iter, total_examples=model.corpus_count)
    print(model.most_similar('embeddings'))

    # TODO: Fix Unking
    print(model.n_similarity(train_abstracts[0].split(), eval_abstracts[1].split()))


def create_tagged_doc(abstracts: [str]):
    return [TaggedDocument(abstract.split(), str(i)) for i, abstract in enumerate(abstracts)]


def main():
    generate_word_embeddings()


if __name__ == '__main__':
    main()
