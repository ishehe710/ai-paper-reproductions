import torch
import torch.nn as nn
from transformer.model.multi_head import MultiHeadAttention
from transformer.model.feed_forward import FeedForwardNetwork
from transformer.model.config import D_MODEL
import math

class DecoderLayer(nn.Module):
    
    def __init__(self, d_model=D_MODEL):
        
        super().__init__()
        
        # layers
        self.attention_layer1 = MultiHeadAttention()
        self.attention_layer2 = MultiHeadAttention()
        self.layernorm_1 = nn.LayerNorm(d_model)
        self.layernorm_2 = nn.LayerNorm(d_model)
        self.layernorm_3 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForwardNetwork()
                
    
    def forward(self, query, key, value):
        
        # masked attention
        mask = self.create_mask(query.size(1), query.device)
        attention_output = self.attention_layer1(query, query, query, mask)
        
        # skip connection and layer norm
        output = attention_output + query
        output = self.layernorm_1(output)
        
        # cross attention
        attention_output = self.attention_layer2(output, key, value)
        
        # skip connection and layer norm
        output = attention_output + output
        output = self.layernorm_2(output)
        
        feed_output = self.feed_forward(output)
        
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