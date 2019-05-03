import json
import sys
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm


def bert(abstract):
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
    tokens_tensor = tokens_tensor.to('cuda')
    model.to('cuda')

    # Predict hidden states features for each layer
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor)
    # We have a hidden states for each of the 12 layers in model bert-base-uncased
    print(encoded_layers[11])
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
    for abstract in tqdm(train_abstracts):
        word_embedding = bert(abstract)
        word_embeddings_train.append(word_embedding)

    for i, abstract in tqdm(enumerate(eval_abstracts)):
        word_embedding_eval = bert(abstract)
        rankings = []
        for train_index, word_embedding_train in enumerate(word_embeddings_train):
            score = dot(word_embedding_eval, word_embedding_train) / (
                    norm(word_embedding_eval) * norm(word_embedding_train))
            rankings.append((score, train_index))

        rankings.sort(key=lambda x: x[0], reverse=True)

        # TODO: now do MRR
        out_citations = eval_out_citations[i]
        if len(out_citations):
            # gets the rankings of the training papers in the correct order
            ranking_ids = get_ids(rankings, train_ids)
            true_citations = [citation for citation in ranking_ids if citation in out_citations]

            if len(true_citations):
                matching_citation_count += 1
                rank = ranking_ids.index(true_citations[0]) + 1
                min_rank = min(min_rank, rank)
                eval_score.append(1.0 / rank)

    print("matching citation count = " + str(matching_citation_count))
    print(eval_score)
    print("min rank = " + str(min_rank))
    print(sum(eval_score) / matching_citation_count)


def split_data(data, dev_start: float, test_start: float, is_test: bool):
    return (data[:int(dev_start * len(data))],
            data[int(test_start * len(data)):] if is_test
            else data[int(dev_start * len(data)): int(test_start * len(data))])


def extract_keys(lines, key: str):
    return [json[key] for json in lines]


def get_ids(rankings, train_ids):
    return [train_ids[index] for _, index in rankings]


def main():
    if len(sys.argv) < 1:
        print('Please supply dataset as command line argument')
        return
    generate_word_embeddings(sys.argv[1])


if __name__ == '__main__':
    main()
