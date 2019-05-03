import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenized input
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = tokenizer.tokenize(text)
print(tokenized_text)

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])

print(tokens_tensor)

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

