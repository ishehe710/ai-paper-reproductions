from transformer.model.config import DATASET_FILENAME, VAL_PERCENTAGE, TEST_PERCENTAGE, BATCH_SIZE
from transformer.data.dataloader import DataLoader, create_dataloader
from transformer.data.dataset import Dataset, create_dataset
import torch
from torch.utils.data import random_split 
from transformer.model.transformer import Transformer

# load dataset
dataset = create_dataset(DATASET_FILENAME)
num_dataset = len(dataset)
print("len(dataset) =", num_dataset)


# make training, validation, and test datasets
num_test = int(TEST_PERCENTAGE * num_dataset)
rest = num_dataset - num_test
num_val = int(VAL_PERCENTAGE * rest)
num_train = rest - num_val

test_ds = [dataset[i] for i in range(num_test)]
rest = [dataset[i] for i in range(num_test, num_dataset)]


train_ds, val_ds = random_split(rest, [num_train, num_val])

'''
print("len(train_ds) =", len(train_ds))
print("len(val_ds)   =", len(val_ds))
print("len(test_ds)  =", len(test_ds))
'''

# create dataloaders
train_loader = create_dataloader(train_ds, BATCH_SIZE, shuffle=True)
val_loader = create_dataloader(val_ds, BATCH_SIZE, shuffle=True)
test_loader = create_dataloader(test_ds, BATCH_SIZE, shuffle=False)

# model preparation


