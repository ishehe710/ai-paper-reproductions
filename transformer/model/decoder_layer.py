import torch
import torch.nn as nn
from transformer.model.multi_head import MultiHeadAttention
from transformer.model.feed_forward import FeedForwardNetwork
from transformer.model.config import D_MODEL, DROPOUT
import math

class DecoderLayer(nn.Module):
    
    def __init__(self, d_model=D_MODEL, dropout=DROPOUT):
        
        super().__init__()
        
        # layers
        self.attention_layer1 = MultiHeadAttention(d_model=d_model)
        self.attention_layer2 = MultiHeadAttention(d_model=d_model)
        self.layernorm_1 = nn.LayerNorm(d_model)
        self.layernorm_2 = nn.LayerNorm(d_model)
        self.layernorm_3 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForwardNetwork(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)
                
    
    def forward(self, query, key, value):
        
        # masked attention
        mask = self.create_mask(query.size(1), query.device)
        attention_output = self.attention_layer1(query, query, query, mask)
        attention_output = self.dropout(attention_output)
        
        # skip connection and layer norm
        output = attention_output + query
        output = self.layernorm_1(output)
        
        # cross attention
        attention_output = self.attention_layer2(output, key, value)
        attention_output = self.dropout(attention_output)
        
        # skip connection and layer norm
        output = attention_output + output
        output = self.layernorm_2(output)
        
        feed_output = self.feed_forward(output)
        feed_output = self.dropout(feed_output)
        
        # skip connection and layer norm
        output = feed_output + output
        output = self.layernorm_3(output)
        
        return output
        
    
    
    def create_mask(self, mask_size, device):
        
        mask = torch.triu(
            torch.full(
                size=(mask_size, mask_size), 
                fill_value=-math.inf,
                device=device
            ),
            diagonal=1
        )
        
        return mask