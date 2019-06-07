# GatesNLP
CSE 481 N: NLP Capstone Project

Not officially sponsored by Gates, only inspired :) 

## Setup
- download our [dataset](https://www.kaggle.com/mitalipalekar/gatesnlp) from Kaggle.
- create the GatesNLP environment (env_linux)
- activate GatesNLP
- run `python -m spacy download en_core_web_sm`
- run `pip install allennlp`

## Run
Run from project root. Ex: `python {PROJECT_ROOT}/src/*.py`

## Evaluate and analyze rankings
Run src/evaluate.py to get a full log of ranking stats or just scores (for using an AllenNLP model,
just input the name of the model as the model)

Run src/sample_ranking_examples.py to randomly sample examples from ranking logs

Notes: update file paths in both files as appropriate, and refer to command-line help

## AllenNLP for Pairwise Model

Run to train

`allennlp train configs/pairs.jsonnet -s [serialization_dir] --include-package gatesnlp`

Run to evaluate

`allennlp evaluate [model] [dataset_to_be_evaluated] --include-package gatesnlp`

Run to predict

`allennlp predict [model] [dataset_to_predict] --include-package gatesnlp --predictor relevance_predictor`