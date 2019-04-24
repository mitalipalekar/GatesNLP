#! /usr/bin/env python3
import wget
import os
import gzip
import glob
import json


BASE_URL = 'https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/'
OUTPUT = 'dataset_final'

def get_corpus_files(filename):
    return [line.strip() for line in filename if line.startswith('corpus')]


def main():
    manifest_file = open('manifest.txt', "r")
    corpus_files = get_corpus_files(manifest_file)
    with open(OUTPUT, 'a+') as out:
        # Reads one file for now, fix bugs and then do all
        for corpus_file in corpus_files:
            wget.download(BASE_URL + corpus_file)
            name = corpus_file[corpus_file.index('/') + 1:]
            with gzip.open(name) as f:
                i = 0
                for line in f:
                    if i % 500 == 0:
                        print(str(i) + ' processed')
                    i = i + 1
                    parsed_json = json.loads(line)
                    if parsed_json['venue'] in ['ACL', 'NAACL','EMNLP']:
                        out.write(line.decode())
							

            # remove files after done
            for f in glob.glob("s2*"):
                os.remove(f)
		    



if __name__ == '__main__':
    main()

