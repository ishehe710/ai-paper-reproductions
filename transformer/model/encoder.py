import torch
import torch.nn as nn
from transformer.model.encoder_layer import EncoderLayer

class Encoder(nn.Module):
    
    def __init__(self, num_layers, d_model):
        
        super().__init__()
        
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model=d_model) for _ in range(num_layers)]
        )
        
        
    def forward(self, x):
        
        output = x
        
        for encoder_layer in self.encoder_layers:
            output = encoder_layer(output)
            
        return output
            
        