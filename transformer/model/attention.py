import torch
import torch.nn as nn
import numpy as np
from transformer.model.positional import output
from transformer.model.config import D_MODEL

class Attention(nn.Module):
    
    def __init__(self, d_model, d_k):
        
        super().__init__()
        
        # query, key, and value layer
        self.query_layer = nn.Linear(in_features=d_model, out_features=d_k)
        self.key_layer = nn.Linear(in_features=d_model, out_features=d_k)
        self.value_layer = nn.Linear(in_features=d_model, out_features=d_k)
        
        # key dimensional for scaling
        self.d_k = d_k
        
        # softmax layer
        self.softmax_layer = nn.Softmax(dim=2)
        
    
    def forward(self, x):
        
        # query, key, and value matrices
        Q = self.query_layer(x)
        K = self.key_layer(x)
        V = self.value_layer(x)
        
        
        K_t = torch.transpose(K, 1, 2)
        result = Q @ K_t # matrix mult. of query and key matrix
        result /= np.sqrt(self.d_k)
        attention_weights = self.softmax_layer(result)
        result = attention_weights @ V
        
        return result
    
    
    
attention_layer = Attention(d_model=D_MODEL, d_k=D_MODEL)

output = attention_layer(output)