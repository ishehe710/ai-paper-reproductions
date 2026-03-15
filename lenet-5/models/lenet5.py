"""
LeNet-5 Implementation in PyTorch

This module implements a reproduction of the classic LeNet-5 convolutional
neural network architecture proposed by Yann LeCun et al. in the paper:

"Gradient-Based Learning Applied to Document Recognition" (1998)

The implementation follows the original architecture while also allowing
modern variations to be tested through configurable parameters.

Architecture Overview
---------------------
Input: 1 × 28 × 28 grayscale image (MNIST)

    C1: Convolution (1 → 6, kernel=5×5) + activation
    S2: Subsampling (AvgPool 2×2)
    C3: Convolution (6 → 16, kernel=5×5) + activation
    S4: Subsampling (AvgPool 2×2)
    C5: Convolution (16 → 120, kernel=5×5)
    F6: Fully Connected (120 → 84)
    Output Layer: RBF classifier or Linear + Softmax

The original LeNet-5 used:
    - Scaled tanh activations
    - Subsampling/Average pooling
    - Radial Basis Function (RBF) classifier

Experimental Variants
---------------------
This implementation allows experimentation with modern alternatives:

    - Activation functions (Scaled Tanh, ReLU)
    - Pooling layers (AvgPool, MaxPool)
    - Optimizers (SGD, Adam)
    - Output classifiers (RBF, Linear + Softmax)
    - Batch Normalization

These experiments are used to compare historical CNN design choices
with commonly used modern deep learning components.

Dataset
-------
MNIST handwritten digit dataset.

Framework
---------
PyTorch

Author
------
Ilagaba Shehe
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 10-element patterns
raw_patterns = [
    [-1, -1, -1, -1, -1,  1,  1,  1,  1,  1], # 0
    [-1,  1, -1,  1, -1,  1, -1,  1, -1,  1], # 1
    [-1, -1,  1,  1, -1, -1,  1,  1, -1, -1], # 2
    [ 1,  1, -1, -1,  1,  1, -1, -1,  1,  1], # 3
    [-1, -1, -1, -1,  1,  1, -1, -1, -1, -1], # 4
    [-1, -1, -1, -1,  1,  1,  1, -1, -1,  1], # 5
    [ 1,  1,  1,  1, -1, -1, -1,  1,  1, -1], # 6
    [-1, -1, -1,  1,  1,  1,  1, -1, -1, -1], # 7
    [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1], # 8
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  # 9
]

# Expand each pattern to length 84 by repeating it
# (10 elements * 9 repetitions = 90 elements, then slice to 84)
expanded_vectors = [torch.tensor(p).repeat(9)[:84] for p in raw_patterns]

# lr: learning rate
def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    """Train the model using gradient descent"""
    history = []
    optimizer = opt_func(model.parameters(), lr=lr)
    # Add a scheduler to help the RBF layer converge
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    
    for epoch in range(epochs):
        model.train()
        # Training Phase
        for batch in train_loader:
            loss = model.training_step(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Validation phase
        scheduler.step() # Decay the learning rate
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history

# define accuracy function
def accuracy(outputs, labels, classifier="rbf"):
    if classifier == "rbf":
        preds = torch.argmin(outputs, dim=1)
    else:
        preds = torch.argmax(outputs, dim=1)
    return (preds == labels).float().mean()

def mle_loss(distances, labels):
    """
    Maximum Likelihood Estimation loss for LeNet-5 style RBF output.

    distances: (batch_size, num_classes)
    labels:    (batch_size,)
    """

    # distance to the correct class
    correct = distances.gather(1, labels.unsqueeze(1)).squeeze(1)

    # competitive term
    competitive = torch.logsumexp(-distances, dim=1)

    # final loss
    loss = correct + competitive

    return loss.mean()

class LeNet5(nn.Module):
    """Feedforward neural network with 1 hidden layer"""
    def __init__(self, activation="tanh", pooling="avg", classifier="rbf", batchnorm=False):
        super().__init__()
        
        # toggling detials
        self.activation_type = activation
        self.activation = None
        self.pooling_type = pooling
        self.classifier_type = classifier
        self.use_batchnorm = batchnorm
        
        # layers
        # convolution layer C1
        self.C1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5))
        
        # convolution layer C3
        self.C3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5))  
        
        # convolution layer C5
        self.C5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5,5))
        
        # full connected layer F6
        self.F6 = nn.Linear(in_features=120, out_features=84) 
        
        # batchnorm 
        self.bn1 = None
        self.bn3 = None
        if batchnorm:
            self.bn1 = nn.BatchNorm2d(6)
            self.bn3 = nn.BatchNorm2d(16)
        else:
            self.bn1 = None
            self.bn3 = None
        
        if activation == "relu":
            self.activation = nn.ReLU()
        
        # layers S2 & S4: either max or avg pooling
        if pooling == "max":
            self.S2 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
            self.S4 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        else:
            self.S2 = nn.AvgPool2d(kernel_size=(2,2), stride=2)
            self.S4 = nn.AvgPool2d(kernel_size=(2,2), stride=2)
            
        if classifier == "rbf":
            self.classifier = RadialBF(torch.stack(expanded_vectors))
        else:
            self.classifier = nn.Linear(84,10)
        
        
    def activate(self, x):
        if self.activation_type == "relu":
            return self.activation(x)
        else:
            # A = 1.7159
            # S = 2/3
            A = 1.7159
            S = 2/3 
            return A*torch.tanh(S*x)
        
    # image is padded            
    def forward(self, images):
        
        # convolution layer 1
        c1_out = self.C1(images)
        
        if self.bn1 is not None:
            c1_out = self.bn1(c1_out)
        
        c1_out = self.activate(c1_out)
        
        # pooling layer 2
        s2_out = self.S2(c1_out)
        
        # convolution layer 3
        c3_out = self.C3(s2_out)
        
        if self.bn3 is not None:
            c3_out = self.bn3(c3_out)
        
        c3_out = self.activate(c3_out)
        
        # pooling layer 4
        s4_out = self.S4(c3_out)
        
        # convolution layer 5
        c5_out = self.activate(self.C5(s4_out))
        
        f6_out = self.activate(self.F6(c5_out.view(images.size(0), 120)))
        
        out = self.classifier(f6_out)
        
        return out
    
    def training_step(self, batch):
        """Returns the loss for a batch of training data"""
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = None
        if self.classifier_type == "rbf":
            loss = mle_loss(out, labels)
        else:
            loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = None
        if self.classifier_type == "rbf":
            loss = mle_loss(out, labels)
        else:
            loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels, self.classifier_type)
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))

# custom Radial Basis Function (RBF) layer
class RadialBF(nn.Module):
    def __init__(self, references):
        super().__init__()
        self.register_buffer("reference_matrix", references)

    def forward(self, x):
        # compute y
        W = self.reference_matrix
        y = ((x.unsqueeze(1) - W)**2).sum(dim=2)
        return y

def evaluate(model, val_loader):
    model.eval()
    with torch.no_grad():
        outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)
