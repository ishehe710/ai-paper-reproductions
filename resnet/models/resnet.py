import torch
from torch import nn
from models.blocks import BasicBlock

class ResNet(nn.Module):
    
    def __init__(self, block, layers, num_classes=1000, stem_type='imagenet', use_skip=True):
        super().__init__()
        
        self.in_channels = 64
        self.stem_type = stem_type
        
        # stem (default to imagenet style)
        kernel_size = 7
        stride = 2
        padding = 3
        
        # choose the indicated stem
        if stem_type == 'cifar10':
            kernel_size = 3
            stride = 1
            padding = 1
            
        # disabling skip connections flag
        self.use_skip = use_skip
            
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # layers
        self.layer1 = self._make_layer(block, out_channels=64, blocks=layers[0], stride=1)
        self.layer2 = self._make_layer(block, out_channels=128, blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(block, out_channels=256, blocks=layers[2], stride=2)
        self.layer4 = self._make_layer(block, out_channels=512, blocks=layers[3], stride=2)
        
        # global average pool and final fully connected layer
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fully_connected = nn.Linear(in_features=512, out_features=num_classes)
    
    
    def _make_layer(self, block, out_channels, blocks, stride):
        layers = []

        # first block may change spatial size / channels
        layers.append(
            block(self.in_channels, out_channels, stride=stride, use_skip=self.use_skip)
        )

        # after first block, channel size is now out_channels
        self.in_channels = out_channels

        # remaining blocks keep same channels and stride=1
        for _ in range(1, blocks):
            layers.append(
                block(self.in_channels, out_channels, stride=1)
            )

        return nn.Sequential(*layers)
    
    def forward(self, x):
        # stem
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if not self.stem_type == 'cifar10':
            out = self.maxpool(out)
        
        # layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # final stage
        out = self.global_avgpool(out)
        out = torch.flatten(out, 1) # flatten for fully connected layer
        out = self.fully_connected(out)
        
        return out

    
def fit(model, num_epochs, train_loader, val_loader, criterion, optimizer, scheduler,device):
    history = {'training': [], 'validation': []}
    for epoch in range(num_epochs):

        model.train()

        # training loop
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # forward
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # training metrics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # Validation and tracking metrics phase
        train_loss = running_loss / total
        train_acc = correct / total
        val_result = evaluate(model, val_loader, criterion, device)
        val_loss, val_acc = val_result
        scheduler.step()
        
        # print epoch metric info
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"\tTrain   Loss: {train_loss:.4f} | Train   Acc: {train_acc:.4f}")
        print(f"\tVal     Loss: {val_loss:.4f}   | Val     Acc: {val_acc:.4f}")
        
        # log the metric data
        history['training'].append((train_loss, train_acc))
        history['validation'].append(val_result)
    
    return history

       

def evaluate(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc

'''
    Versions of the ResNet Model with varying amount of layers.
'''

def resnet18(num_classes=1000, stem_type='imagenet', use_skip=True):
    '''
    Facotry method for the ResNet-18, model with .
    '''
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, stem_type=stem_type, use_skip=use_skip)