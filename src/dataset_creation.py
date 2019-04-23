#! /usr/bin/env python3

def get_corpus_files(filename):
    files = []
    for line in filename:
        print(line)
        if line.startswith('corpus'):
            files.append(line.rstrip("\n"))
    return files

def main():
    manifest_file = open('manifest.txt', "r")
    corpus_files = get_corpus_files(manifest_file)
    print(corpus_files)

if __name__ == '__main__':
    main()

