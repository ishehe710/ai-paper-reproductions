import torch
import torch.nn as nn
from transformer.model.multi_head import MultiHeadAttention
from transformer.model.config import D_MODEL, DROPOUT
from transformer.model.feed_forward import FeedForwardNetwork

class EncoderLayer(nn.Module):
    
    def __init__(self, d_model=D_MODEL, dropout=DROPOUT):
        
        super().__init__()
        
        # layers
        self.attention_layer = MultiHeadAttention(d_model=d_model)
        self.layernorm_1 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForwardNetwork(d_model=d_model)
        self.layernorm_2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        
    
    def forward(self, x):
        
        attention_output = self.attention_layer(x, x, x)
        attention_output = self.dropout(attention_output)
        
        # residual connection
        output = attention_output + x
        output = self.layernorm_1(output)
        
        feed_forward_output = self.feed_forward(output)
        
        # residual connection        
        output = feed_forward_output + output
        output = self.dropout(output)
        output = self.layernorm_2(output)
        
        return output
        
        