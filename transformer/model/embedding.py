import torch
import torch.nn as nn
from transformer.model.config import D_MODEL
from transformer.data.dataloader import english_df, src_batch
from transformer.data.vocabulary import map_token_to_id
from transformer.data.tokenizer import tokenize

english_vocab = map_token_to_id(tokenize(english_df))


embedding_layer = nn.Embedding(num_embeddings=len(english_vocab), embedding_dim=D_MODEL)

print(src_batch.shape)
output = embedding_layer(src_batch)
print(output.shape)
