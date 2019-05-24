import json

from src.gnlputils import extract_keys

GATESNLP_ROOT = '/projects/instr/19sp/cse481n/GatesNLP/'
PAPERS = GATESNLP_ROOT + 'extended_dataset.txt'
CORPUS = GATESNLP_ROOT + 'glove_corpus.txt'


def main():
    with open(PAPERS, 'rb') as f:
        lines = [json.loads(line) for line in f]
    abstracts = extract_keys(lines, 'paperAbstract')
    titles = extract_keys(lines, 'title')

    text = [(t + " " + a).replace('\n', ' ') for t, a in zip(titles, abstracts)]
    with open(CORPUS, 'w') as f:
        f.writelines("{}\n".format(document) for document in text)


if __name__ == '__main__':
    main()
