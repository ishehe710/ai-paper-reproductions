import torch
import torchvision
from torchvision.transforms import v2
from torchvision.datasets import MNIST
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

# in-project imports
from gan.models.config import DATA_FLODER_PATH, VAL_SET_LENGTH, NUM_WORKERS, BATCH_SIZE

# load dataset
transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

train_dataset = MNIST(root=DATA_FLODER_PATH, train=True, download=True, transform=transform)
test_dataset = MNIST(root=DATA_FLODER_PATH, train=False, download=True, transform=transform)

# create dataloaders
train_length = len(train_dataset) - VAL_SET_LENGTH
train_ds, val_ds = random_split(train_dataset, [train_length, VAL_SET_LENGTH])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


# sample

batch = next(iter(train_loader))