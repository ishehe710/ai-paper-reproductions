import torch
import torch.nn as nn
from transformer.model.config import D_MODEL 
from transformer.model.attention import Attention
from transformer.model.positional import output

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads=8, d_model=D_MODEL):
        
        assert d_model % num_heads == 0
        super().__init__()
        
        # necessary values
        self.head_dim = d_model // num_heads
        self.num_heads = num_heads
        
        # layers
        self.W_query_layer = nn.Linear(in_features=d_model, out_features=d_model)
        self.W_key_layer = nn.Linear(in_features=d_model, out_features=d_model)
        self.W_value_layer = nn.Linear(in_features=d_model, out_features=d_model)
        self.W_output_layer = nn.Linear(in_features=d_model, out_features=d_model)
        
        # attention layers
        self.attention_layers = nn.ModuleList(
            [Attention(d_k=self.head_dim) for _ in range(num_heads)]
        )
        
        
        
    
    def forward(self, query_x, key_x, value_x, mask=None):
        
        W_q = self.W_query_layer(query_x)
        W_k = self.W_key_layer(key_x)
        W_v = self.W_value_layer(value_x)
                
        # split the into heads for the attention
        head_outputs = [0] * self.num_heads
        for i in range(self.num_heads):
            attention_layer = self.attention_layers[i]
            start = self.head_dim * i
            end = self.head_dim * (i+1) 
            head_outputs[i] = attention_layer(
                W_q[:, :, start:end],
                W_k[:, :, start:end],
                W_v[:, :, start:end],
                mask=mask
            )
        
        # concat 
        result = torch.cat(tuple(head_outputs), dim=2)

        result = self.W_output_layer(result)
        return result
    
    