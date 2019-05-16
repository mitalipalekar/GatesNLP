import json
import pickle
import sys

import torch
from gnlputils import extract_keys, split_data, cosine_similarity
from gnlputils import get_from_rankings
from pytorch_pretrained_bert import BertTokenizer, BertModel
from tqdm import tqdm

WORD_EMBEDDINGS_TRAIN = 'complete_bert_embeddings_train_fix_bugs_v2.pk'
WORD_EMBEDDINGS_EVAL = 'complete_bert_embeddings_eval_fix_bugs_v2.pk'


def take_mean_bert(vector):
    return torch.mean(vector[0], dim=0)


def bert(abstract, tokenizer, model):
    # Tokenized input
    tokenized_text = tokenizer.tokenize(abstract)
    if len(tokenized_text) > 512:
        tokenized_text = tokenized_text[:512]

    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = get_segment_ids(tokenized_text)
    assert len(indexed_tokens) == len(segments_ids)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensor = torch.tensor([segments_ids])

    # If you have a GPU, put everything on cuda
    if torch.cuda.is_available():
        with torch.cuda.device(1): # NOTE: what you wanna change to set device of GPU
            tokens_tensor = tokens_tensor.cuda()
            segments_tensor = segments_tensor.cuda()
            model = model.cuda()

    # Predict hidden states features for each layer
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensor)
    # We have a hidden states for each of the 12 layers in model bert-base-uncased
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
    matching_citation_count = 1
    min_rank = float("inf")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()

    for abstract in tqdm(train_abstracts, desc='Extracting embeddings for training set'):
        with open(WORD_EMBEDDINGS_TRAIN, 'ab') as handle:
            if abstract:
                abstract = abstract.lower()
                word_embedding = take_mean_bert(bert(abstract, tokenizer, model))
                pickle.dump(word_embedding, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for i, abstract in tqdm(enumerate(eval_abstracts), desc='Extracting embeddings for evaluation set'):
        if abstract:
            abstract = abstract.lower()
            word_embedding_eval = take_mean_bert(bert(abstract, tokenizer, model))
            with open(WORD_EMBEDDINGS_EVAL, 'ab') as f:
                pickle.dump(word_embedding_eval, f, protocol=pickle.HIGHEST_PROTOCOL)

            with open(WORD_EMBEDDINGS_TRAIN, 'rb') as handle:
                rankings = []
                try:
                    train_index = 0
                    while True:
                        word_embedding_train = pickle.load(handle)
                        score = cosine_similarity(word_embedding_eval.cpu(), word_embedding_train.cpu())
                        rankings.append((score, train_index))
                        train_index += 1
                except EOFError:
                    handle.seek(0)
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
        break

    print("Matching citation count = {0}".format(str(matching_citation_count)))
    print(eval_score)
    print("Min rank = {0}".format(str(min_rank)))
    print(sum(eval_score) / matching_citation_count)


def get_segment_ids(tokenized_text):
    sentence_ids = []
    i = 0
    for token in tokenized_text:
        sentence_ids.append(i)
        if token == ".":
            i += 1
    return sentence_ids


def main():
    generate_word_embeddings(sys.argv[1])


if __name__ == '__main__':
    main()
