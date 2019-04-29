#! /usr/bin/env python3
import wget
import os
import gzip
import glob
import json
from tqdm import tqdm

BASE_URL = 'https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/'
OUTPUT = 'extended_dataset.txt'
VENUES = ['ACL', 'NAACL', 'EMNLP', 'ACM Conference on Computer and Communications Security', 'IEEE Symposium on Security and Privacy', 'IEEE International Conference on Information Theory and Information Security', 'IEEE Transactions on Information Forensics and Security', 'USENIX Security Symposium']


def main():
    manifest_file = open('manifest.txt', "r")
    corpus_files = get_corpus_files(manifest_file)

    # Remember this is append, make sure your file doesn't exist
    with open(OUTPUT, 'a+') as out:
        for corpus_file in tqdm(corpus_files, desc='Corpus'):
            wget.download(BASE_URL + corpus_file)
            name = corpus_file[corpus_file.index('/') + 1:]
            process_corpus(name, out)
            # remove files after done
            for f in glob.glob("s2*"):
                os.remove(f)


def process_corpus(name, out):
    with gzip.open(name) as f:
        for line in tqdm(f, desc='Line'):
            parsed_json = json.loads(line)
            if substring_of_array(parsed_json['venue']):
                out.write(line.decode())


def get_corpus_files(filename):
    return [line.strip() for line in filename if line.startswith('corpus')]

def substring_of_array(parsed_venue):
    for venue in VENUES:
        if venue in parsed_venue:
            return True;
    return False


if __name__ == '__main__':
    main()
