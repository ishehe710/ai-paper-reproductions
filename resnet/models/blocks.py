from torch import nn

class BasicBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1, use_skip=True):
        super().__init__()
        # convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
            )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        
        # relu activations
        self.relu = nn.ReLU(inplace=True)
        
        # batch norm
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # use skip
        self.use_skip = use_skip
        # skip path
        if stride != 1 or in_channels != out_channels:
            # projection shortcut
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        # addition and final activation
        if self.use_skip:
            out = out + identity
        out = self.relu(out)
        
        return out
        
        
        