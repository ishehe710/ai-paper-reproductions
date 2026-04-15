import torch
from models.blocks import BasicBlock
from models.resnet import resnet18

seed = 42
torch.manual_seed(seed)

'''
    Testing BasicBlock objects
'''
block1 = BasicBlock(64, 64, stride=1)   # regular
block2 = BasicBlock(64, 128, stride=2)  # downsampling

x1 = torch.randn(32, 64, 56, 56)

print("Testing BasicBlock")
print("\tx1.shape =", x1.shape)

y1 = block1(x1)
y2 = block2(x1)

print("\ty1.shape =", y1.shape)
print("\ty2.shape =", y2.shape)
print("\n")

'''
    Testing ResNet-18
'''
model = resnet18(num_classes=10)

print("Testing ResNet-18")

x1 = torch.randn(4, 3, 224, 224)
print("\tx1.shape =", x1.shape)

y1 = model(x1)
print("\ty1.shape =", y1.shape)
print("\n")