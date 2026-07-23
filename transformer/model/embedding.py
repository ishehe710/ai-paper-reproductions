import torch
import torch.nn as nn
import numpy as np
from transformer.model.config import D_MODEL
from transformer.data.dataloader import english_df, src_batch
from transformer.data.vocabulary import map_token_to_id
from transformer.data.tokenizer import tokenize

class TransformerEmbedding(nn.Module):
    
    def __init__(self, vocab_size, d_model):
        super().__init__()  
        self.d_model = d_model
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    
    
    def forward(self, x):
        # 1. Look up raw vectors -> Shape: [batch_size, seq_len, d_model]
        x_embed = self.embedding(x)
        
        # 2. Apply the scale factor
        scaled_embed = x_embed * np.sqrt(self.d_model)
        
        return scaled_embed


vocab_size = len(map_token_to_id(tokenize(english_df)).keys())
embedding_layer = TransformerEmbedding(vocab_size, D_MODEL)
print(src_batch.shape)
output = embedding_layer(src_batch)
print(output.shape)
print(output.dtype)
