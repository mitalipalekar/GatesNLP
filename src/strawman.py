#! /usr/bin/env python3
import pandas as pd

PAPERS = "../dataset/papers.csv"


def main():
    df = pd.read_csv(PAPERS)
    print(df)


if __name__ == '__main__':
    main()
