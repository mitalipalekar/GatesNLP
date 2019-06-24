# GatesNLP: An Experimental Extension to Content-Based Citation Recommendation
CSE 481 N: UW NLP Capstone Project

Not officially sponsored by Gates, only inspired!

This GitHub repo contains source code for content-based citation recommendation research conducted for UW's NLP capstone in Spring 2019.
Below is the resulting report's abstract, and for more information, please read our
full [report](https://github.com/mitalipalekar/GatesNLP/blob/master/GatesNLP_CSE_481_N_Final_Report.pdf).
We provide instructions on how to run our existing code in the hopes that this can be a reference and starting point for
future NLP students and researchers.

For our University of Washington NLP capstone, we present an extension of prior work on content-based citation recommendation. Given a query paper’s text and candidate papers’ text, we generate a ranked list of candidate papers in which cited papers appear first. We experiment with a variety of supervised and unsupervised learning approaches to generate rankings as well as evaluate the effectiveness of our models’ rankings quantitatively and qualitatively. We find that our best model is TF-IDF, achieving a mean reciprocal rank of 0.31 on our test set. Finally, we release two new datasets of natural language processing and computer security papers to facilitate future academic research.

## Setup
- download our [dataset](https://www.kaggle.com/mitalipalekar/gatesnlp) from Kaggle.
- create the GatesNLP environment (env_linux)
- activate GatesNLP
- run `python -m spacy download en_core_web_sm`
- run `pip install allennlp`

## Run
Run from project root. Ex: `python {PROJECT_ROOT}/src/*.py`

The models we provide with our evaluation script (src/evaluate.py) use the following scoring functions for each query/candidate pair:
- Jaccard similarity index
- TF-IDF with cosine similarity
- Confidence score from a supervised citation classifier (given a pair of paper texts, predicting if the first cites the second)

Models in other scripts use these scoring functions:
- BERT with cosine similarity (src/bert/bert_embeddings.py)
- GloVe with cosine similarity and Word Mover's Distance (src/glove/glove_embeddings.py)


## Evaluate and analyze rankings
Run src/evaluate.py to get a full log of ranking stats or just scores (for using an AllenNLP model,
just input the name of the model as the model)

Run src/sample_ranking_examples.py to randomly sample examples from ranking logs

Notes: update file paths in both files as appropriate, and refer to command-line help

## AllenNLP for Supervised Model

Run to train

`allennlp train configs/pairs.jsonnet -s [serialization_dir] --include-package gatesnlp`

Run to evaluate

`allennlp evaluate [model] [dataset_to_be_evaluated] --include-package gatesnlp`

Run to predict

`allennlp predict [model] [dataset_to_predict] --include-package gatesnlp --predictor relevance_predictor`
