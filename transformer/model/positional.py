import torch
import torch.nn as nn
import numpy as np
from transformer.model.config import MAX_LENGTH_ENCODING, D_MODEL


class PositionalEncoding(nn.Module):
    
    def __init__(self, max_length, d_model):
        super().__init__()
        
        pe = torch.zeros(max_length, d_model)
        
        
        for pos in range(max_length):
            for i in range(d_model//2):
                theta = pos/10000**(2*i/d_model)
                pe[pos, 2*i] = np.sin(theta)
                pe[pos, 2*i + 1] = np.cos(theta)
                
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :]
    
    
    
