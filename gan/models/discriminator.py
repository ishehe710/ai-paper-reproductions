import torch
import torch.nn as nn
import torch.functional as F

from gan.models.config import NUM_PIECES, UNITS

class Discriminator(nn.Module):
    
    def __init__(self, num_pieces):
        
        super().__init__()
        
        self.flatten = nn.Flatten()
        
        
        self.mlp_stack = nn.Sequential(
            Maxout(in_features=784, out_features=UNITS, num_pieces=num_pieces),
            Maxout(in_features=UNITS, out_features=UNITS, num_pieces=num_pieces),
            nn.Linear(in_features=UNITS, out_features=1),
            nn.Sigmoid()            
        )
        
    # takes 28x28 images, not flatten
    def forward(self, x):
        
        x = self.flatten(x)
        output = self.mlp_stack(x)
        
        return output
    
    def loss(self, fake_preds, real_preds):
        eps = 1e-8
        real_loss = torch.log(real_preds).mean()
        fake_loss = torch.log(1 - fake_preds).mean()

        return -(real_loss + fake_loss)
        #return -(torch.log(real_preds) + torch.log(1 - fake_preds)).mean()
    

class Maxout(nn.Module):
    def __init__(self, in_features, out_features, num_pieces):
        super(Maxout, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_pieces = num_pieces  # Number of linear pieces to max over
        
        # Multiply out_features by num_pieces to calculate all groups at once
        self.linear = nn.Linear(in_features, out_features * num_pieces)

    def forward(self, x):
        # 1. Shape of input x: (batch_size, in_features)
        raw_output = self.linear(x)
        
        # 2. Reshape to split the last dimension into (out_features, num_pieces)
        # New shape: (batch_size, out_features, num_pieces)
        reshaped = raw_output.view(-1, self.out_features, self.num_pieces)
        
        # 3. Take the maximum value along the piece dimension (dim=-1)
        # torch.max returns a tuple of (values, indices); we only need values
        max_output, _ = torch.max(reshaped, dim=-1)
        
        return max_output

