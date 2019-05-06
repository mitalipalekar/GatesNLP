import json
import pickle
import sys

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
from tqdm import tqdm

from src.gnlputils import extract_keys, split_data

WORD_EMBEDDINGS_TRAIN = 'word_embeddings_train.pk'
WORD_EMBEDDINGS_EVAL = 'word_embeddings_eval.pk'


def bert(abstract):
    # print(len(abstract))
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenized input
    tokenized_text = tokenizer.tokenize(abstract)

    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()

    # If you have a GPU, put everything on cuda
    # tokens_tensor = tokens_tensor.to('cuda')
    # model.to('cuda')

    # Predict hidden states features for each layer
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor)
    # We have a hidden states for each of the 12 layers in model bert-base-uncased
    # print(encoded_layers[11])
    # print(encoded_layers[11].shape)
    return encoded_layers[11]


def generate_word_embeddings(papers):
    lines = []
    with open(papers, 'rb') as f:
        for line in tqdm(f):
            lines.append(json.loads(line))

    lines.sort(key=lambda x: x['year'])

    ids = extract_keys(lines, 'id')
    abstracts = extract_keys(lines, 'paperAbstract')
    titles = extract_keys(lines, 'title')
    out_citations = extract_keys(lines, 'outCitations')

    # TODO: DO NOT HARDCODE THIS
    is_test = False

    train_ids, eval_ids = split_data(ids, 0.8, 0.9, is_test)
    train_abstracts, eval_abstracts = split_data(abstracts, 0.8, 0.9, is_test)
    train_title, eval_title = split_data(titles, 0.8, 0.9, is_test)
    train_out_citations, eval_out_citations = split_data(out_citations, 0.8, 0.9, is_test)

    eval_score = []
    matching_citation_count = 0
    min_rank = float("inf")
    word_embeddings_train = []
    print('--------- extracting embeddings for training set --------')
    # i = 0
    for abstract in tqdm(train_abstracts):
        if abstract:
            word_embedding = bert(abstract)
            word_embeddings_train.append(word_embedding)
        i = i + 1
        # if i == 10:
        #     break
    with open(WORD_EMBEDDINGS_TRAIN, 'wb') as handle:
        pickle.dump(word_embeddings_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('--------- finished extracting embeddings for training set --------')

    print('--------- extracting embeddings for evaluation set --------')
    word_embeddings_eval = []
    for i, abstract in tqdm(enumerate(eval_abstracts)):
        word_embedding_eval = bert(abstract)
        word_embeddings_eval.append(word_embedding_eval)
        # rankings = []
        # for train_index, word_embedding_train in enumerate(word_embeddings_train):
        #     score = dot(word_embedding_eval, word_embedding_train) / (
        #             norm(word_embedding_eval) * norm(word_embedding_train))
        #     rankings.append((score, train_index))
        #
        # rankings.sort(key=lambda x: x[0], reverse=True)
        #
        # # TODO: now do MRR
        # out_citations = eval_out_citations[i]
        # if len(out_citations):
        #     # gets the rankings of the training papers in the correct order
        #     ranking_ids = get_ids(rankings, train_ids)
        #     true_citations = [citation for citation in ranking_ids if citation in out_citations]
        #
        #     if len(true_citations):
        #         matching_citation_count += 1
        #         rank = ranking_ids.index(true_citations[0]) + 1
        #         min_rank = min(min_rank, rank)
        #         eval_score.append(1.0 / rank)
        # break
    with open(WORD_EMBEDDINGS_EVAL, 'wb') as handle:
        pickle.dump(word_embeddings_eval, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('--------- finished extracting embeddings for evaluation set --------')

    # print("matching citation count = " + str(matching_citation_count))
    # print(eval_score)
    # print("min rank = " + str(min_rank))
    # print(sum(eval_score) / matching_citation_count)


def main():
    generate_word_embeddings(sys.argv[1])


if __name__ == '__main__':
    main()
