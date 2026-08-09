import torch
import torch.nn as nn
import torch.functional as F
import matplotlib.pyplot as plt
import numpy as np

from gan.models.config import LATENT_D

class Generator(nn.Module):
    
    def __init__(self, output_bias):
        
        super().__init__()
    
       
        # layers
        
        self.mlp_stack = nn.Sequential(
            nn.Linear(in_features=LATENT_D, out_features=1200),
            nn.ReLU(),
            nn.Linear(in_features=1200, out_features=1200),
            nn.ReLU(),
            nn.Linear(in_features=1200, out_features=784, bias=True),
            nn.Sigmoid()
        )
        
        
        with torch.no_grad():
            self.mlp_stack[-2].bias.copy_(output_bias)
            
        
        
    
    def forward(self, z):
        
        output = self.mlp_stack(z)
        
        return output
    

    def loss(self, D_preds):
        eps = 1e-8
        return torch.log(1 - D_preds).mean()
    
    # ablation 1
    def saturated_loss(self, D_preds):
        return -torch.log(D_preds).mean()

''' 
G = Generator(5)

input = torch.randn(LATENT_D)

print(input)

output = G(input)
print(output.shape)


plt.imshow(output.detach().numpy().reshape(28, 28))
plt.show()
'''