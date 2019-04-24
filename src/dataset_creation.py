#! /usr/bin/env python3
import wget
import os
import gzip
import glob


BASE_URL = 'https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/'
def get_corpus_files(filename):
    return [line.strip() for line in filename if line.startswith('corpus')]


def main():
    manifest_file = open('manifest.txt', "r")
    corpus_files = get_corpus_files(manifest_file)
    print(corpus_files)
    for corpus_file in corpus_files[:1]:
	    wget.download(BASE_URL + corpus_file)
	    name = corpus_file[corpus_file.index('/') + 1:]
	    with gzip.open(name) as f:
		    for line in f:
			    print(line)

            # remove files after done
	    for f in glob.glob("s2*"):
		    os.remove(f)
	    



if __name__ == '__main__':
    main()

