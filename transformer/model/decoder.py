import torch
import torch.nn as nn
from transformer.model.decoder_layer import DecoderLayer
from transformer.model.config import D_MODEL

class Decoder(nn.Module):
    
    def __init__(self, num_layers, d_model):
        
        super().__init__()
        
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model=d_model) for _ in range(num_layers)]
        )
        
    
    def forward(self, x, encoder_output):
        
        output = x
        
        for decoder_layer in self.decoder_layers:
            output = decoder_layer(output, encoder_output, encoder_output)
            
            
        return output