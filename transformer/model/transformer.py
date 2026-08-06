import torch
import torch.nn as nn
from transformer.model.embedding import TransformerEmbedding
from transformer.model.positional import PositionalEncoding
from transformer.model.config import MAX_LENGTH_ENCODING
from transformer.model.encoder import Encoder
from transformer.model.decoder import Decoder

class Transformer(nn.Module):
    
    def __init__(self, source_vocab_size, target_vocab_size, d_model, num_layers, dropout):
        
        super().__init__()
        
        
        # layers
        
        # embeddings
        self.input_embedding = TransformerEmbedding(
            vocab_size=source_vocab_size,
            d_model=d_model
            )
        self.output_embedding = TransformerEmbedding(
            vocab_size=target_vocab_size,
            d_model=d_model
        )
        
        # positional encodings
        self.position_encoding = PositionalEncoding(max_length=MAX_LENGTH_ENCODING, d_model=d_model)
        
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model)
        self.decoder = Decoder(num_layers=num_layers, d_model=d_model)
        
        self.linear_layer = nn.Linear(in_features=d_model, out_features=target_vocab_size)    
        self.dropout = nn.Dropout(p=dropout)    

    def forward(self, source, target):
        
        # encoding half
        embedded_source = self.input_embedding(source)
        summed_source = self.position_encoding(embedded_source)
        summed_source = self.dropout(summed_source)
        encoder_output = self.encoder(summed_source)
        
        # decoding half
        embedded_target = self.output_embedding(target)
        summed_target = self.position_encoding(embedded_target)
        decoder_output = self.decoder(x=summed_target, encoder_output=encoder_output)
        output = self.linear_layer(decoder_output)
        
        return output
        
        