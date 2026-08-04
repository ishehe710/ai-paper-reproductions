from transformer.model.config import DATASET_FILENAME, VAL_PERCENTAGE, TEST_PERCENTAGE, BATCH_SIZE, D_MODEL, NUM_LAYERS, EPOCHS
from transformer.data.dataloader import create_dataloader
from transformer.data.dataset import Dataset, create_dataset
import torch
from torch.utils.data import random_split 
from transformer.model.transformer import Transformer
from transformer.model.train import fit
import torch.nn as nn
import torch.optim as optim
from transformer.model.evaluate import evaluate

# 1. load dataset
dataset, eng_vocab_size, frc_vocab_size = create_dataset(DATASET_FILENAME)
num_dataset = len(dataset)
print("len(dataset) =", num_dataset)




# make training, validation, and test datasets
num_test = int(TEST_PERCENTAGE * num_dataset)
rest = num_dataset - num_test
num_val = int(VAL_PERCENTAGE * rest)
num_train = rest - num_val



train_ds, val_ds, test_ds = random_split(dataset, [num_train, num_val, num_test])


# create dataloaders
train_loader = create_dataloader(train_ds, BATCH_SIZE, shuffle=True)
val_loader = create_dataloader(val_ds, BATCH_SIZE, shuffle=True)
test_loader = create_dataloader(test_ds, BATCH_SIZE, shuffle=False)






# 2. model preparation



model = Transformer(
    source_vocab_size=eng_vocab_size,
    target_vocab_size=frc_vocab_size,
    d_model=D_MODEL,
    num_layers=NUM_LAYERS
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 3. training set up

# cross entropy loss function
loss = nn.CrossEntropyLoss()


# Adam optimizer
lrate = 0.0001

optimizer = optim.Adam(
    model.parameters(),
    lr=lrate
)

scheduler = None

# 4. training


subset_train = torch.utils.data.Subset(train_ds, range(4096))
subset_val = torch.utils.data.Subset(val_ds, range(512))
subset_test = torch.utils.data.Subset(test_ds, range(512))

train_loader = create_dataloader(subset_train, BATCH_SIZE)
val_loader = create_dataloader(subset_val, BATCH_SIZE)
test_loader = create_dataloader(subset_test, BATCH_SIZE)


print(device)
print(torch.cuda.is_available())


num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params:,}")

history = fit(
    model=model, 
    num_epochs=EPOCHS,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_func=loss,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device    
)


print(history)

# 5. test
evaluate(model=model, dataloader=test_loader, loss_func=loss, device=device)



