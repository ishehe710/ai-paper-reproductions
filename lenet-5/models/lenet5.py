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
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        model.train()
        # Training Phase
        for batch in train_loader:
            loss = model.training_step(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history

# define accuracy function
def accuracy(outputs, labels):
    preds = torch.argmin(outputs, dim=1)
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
    loss = correct - competitive

    return loss.mean()

class LeNet5(nn.Module):
    """Feedforward neural network with 1 hidden layer"""
    def __init__(self):
        super().__init__()
        # convolution layer C1
        self.C1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5))
        
        # subsampling layer S2
        self.S2 = nn.AvgPool2d(kernel_size=(2,2), stride=2)
        
        # convolution layer C3
        self.C3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5))

        # subsampling layer S4
        self.S4 = nn.AvgPool2d(kernel_size=(2,2), stride=2)    
        
        # convolution layer C5
        self.C5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5,5))
        
        # full connected layer F6
        self.F6 = nn.Linear(in_features=120, out_features=84) 
        
        # output layer (RBF)
        self.rbf = RadialBF(torch.stack(expanded_vectors))
        
    # image is padded            
    def forward(self, images):
        # activation function
        # A = 1.7159
        # S = 2/3
        A = 1.7159
        S = 2/3 
        
        c1_out = A*torch.tanh(S*self.C1(images))
        s2_out = self.S2(c1_out)
        c3_out = A*torch.tanh(S*self.C3(s2_out))
        s4_out = self.S4(c3_out)
        c5_out = A*torch.tanh(S*self.C5(s4_out))
        f6_out = A*torch.tanh(S*self.F6(c5_out.view(images.size(0), 120)))
        out = self.rbf(f6_out)
        
        return out
    
    def training_step(self, batch):
        """Returns the loss for a batch of training data"""
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = mle_loss(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = mle_loss(out, labels)
        acc = accuracy(out, labels)         # Calculate accuracy
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

