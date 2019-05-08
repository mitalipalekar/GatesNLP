import json
import pickle
import sys

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
from tqdm import tqdm

from gnlputils import extract_keys, split_data, cosine_similarity

from gnlputils import get_from_rankings

WORD_EMBEDDINGS_TRAIN = 'word_embeddings_train.pk'
WORD_EMBEDDINGS_EVAL = 'word_embeddings_eval.pk'


def take_mean_bert(vector):
    return torch.mean(vector[0], dim=0)


def bert(abstract):
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenized input
    tokenized_text = tokenizer.tokenize(abstract)
    if len(tokenized_text) > 512:
        tokenized_text = tokenized_text[:512]

    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()

    # If you have a GPU, put everything on cuda
    tokens_tensor = tokens_tensor.to('cuda')
    model.to('cuda')

    # Predict hidden states features for each layer
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor)
    # We have a hidden states for each of the 12 layers in model bert-base-uncased
    return encoded_layers[11end(word_embedding)
    with open(WORD_EMBEDDINGS_TRAIN, 'wb') as handle:
        pickle.dump(word_embeddings_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

    word_embeddings_eval = []
    for i, abstract in tqdm(enumerate(eval_abstracts), desc='extracting embeddings for evaluation set'):
        word_embedding_eval = take_mean_bert(bert(abstract))
        word_embeddings_eval.append(word_embedding_eval)
        rankings = []
        for train_index, word_embedding_train in enumerate(word_embeddings_train):
            score = cosine_similarity(word_embedding_eval.cpu(), word_embedding_train.cpu())
            rankings.append((score, train_index))
        rankings.sort(key=lambda x: x[0], reverse=True)

        out_citations = eval_out_citations[i]
        if len(out_citations):
            # gets the rankings of the training papers in the correct order
            ranking_ids = get_from_rankings(rankings, train_ids)
            true_citations = [citation for citation in ranking_ids if citation in out_citations]

            if len(true_citations):
                matching_citation_count += 1
                rank = ranking_ids.index(true_citations[0]) + 1
                min_rank = min(min_rank, rank)
                eval_score.append(1.0 / rank)
    with open(WORD_EMBEDDINGS_EVAL, 'wb') as handle:
        pickle.dump(word_embeddings_eval, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("matching citation count = " + str(matching_citation_count))
    print(eval_score)
    print("min rank = " + str(min_rank))
    print(sum(eval_score) / matching_citation_count)


def main():
    generate_word_embeddings(sys.argv[1])


if __name__ == '__main__':
    main()
