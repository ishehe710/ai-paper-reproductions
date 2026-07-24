import torch
import torch.nn as nn
import numpy as np
from transformer.model.positional import output
from transformer.model.config import D_MODEL

class Attention(nn.Module):
    
    def __init__(self, d_model, d_k):
        
        super().__init__()
        
        
        # key dimensional for scaling
        self.d_k = d_k
        
        # softmax layer
        self.softmax_layer = nn.Softmax(dim=2)
        
    
    def forward(self, query, key, value):
        
        # query, key, and value matrices
        Q = query
        K = key
        V = value
        
        
        K_t = torch.transpose(K, 1, 2)
        result = Q @ K_t # matrix mult. of query and key matrix
        result /= np.sqrt(self.d_k)
        attention_weights = self.softmax_layer(result)
        result = attention_weights @ V
        
        return result
    
    
    