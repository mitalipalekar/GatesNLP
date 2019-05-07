#! /usr/bin/env python3
import json
from tqdm import tqdm

PAPERS = '/projects/instr/19sp/cse481n/GatesNLP/extended_dataset.txt'
OUTPUT = '/projects/instr/19sp/cse481n/GatesNLP/pairs.txt'


def main():
    f = open(PAPERS, 'r')
    out = open(OUTPUT, 'w')

    text = dict()
    outCitations = dict()
    for line in f:
        paper = json.loads(line)
        id = paper['id']
        text[id] = paper['title'] + ' ' + paper['paperAbstract']
        outCitations[id] = paper['outCitations']

    # cited pairs
    for paper1, text1 in tqdm(text.items()):
        for paper2 in outCitations[paper1]:
            if paper2 in text.keys():
                result = (text[paper2] + "\t" + text1 + "\t1").encode("unicode_escape").decode("utf-8") + "\n"
                out.write(result)

    # one-hop pairs
    for paper1, text1 in tqdm(text.items()):
        seen = set()
        for paper2 in outCitations[paper1]:
            if paper2 in text.keys():
                for paper3 in outCitations[paper2]:
                    if paper3 in text.keys() and paper3 not in seen and paper3 not in outCitations[paper1]:
                        seen.add(paper3)
                        result = (text[paper3] + "\t" + text1 + "\t0").encode("unicode_escape").decode("utf-8") + "\n"
                        out.write(result)


if __name__ == '__main__':
    main()
