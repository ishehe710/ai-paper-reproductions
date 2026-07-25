import torch
import torch.nn as nn
import math 
from transformer.model.positional import output
from transformer.model.config import D_MODEL

class Attention(nn.Module):
    
    def __init__(self, d_k):
        
        super().__init__()
        
        
        # key dimensional for scaling
        self.d_k = d_k
        
        # softmax layer
        self.softmax_layer = nn.Softmax(dim=2)
        
    
    def forward(self, query, key, value, mask=None):
        
        # query, key, and value matrices
        Q = query
        K = key
        V = value
        
        
        K_t = torch.transpose(K, 1, 2)
        result = Q @ K_t # matrix mult. of query and key matrix
        result /= math.sqrt(self.d_k)
        if mask is not None:
            result += mask
        attention_weights = self.softmax_layer(result)
        result = attention_weights @ V
        
        return result
    
    
    