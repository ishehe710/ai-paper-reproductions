import torch
import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    
    def __init__(self, d_model=128, d_ff=512):
        super().__init__()
        
        # layers
        self.layer_1 = nn.Linear(in_features=d_ff, out_features=d_model)
        self.layer_2 = nn.Linear(in_features=d_model, out_features=d_ff)
        self.relu = nn.ReLU()
        
    
    def forward(self, x):
        
        result = self.layer_1(x)
        result = self.relu(result)
        result = self.layer_2(x)
        
        return result
    
    
feed_forward = FeedForwardNetwork()

    