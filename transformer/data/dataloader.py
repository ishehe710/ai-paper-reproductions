import pandas as pd
import torch
from transformer.data.dataset import create_id_seqs, TranslationDataset
from torch.utils.data import Dataset, DataLoader
from transformer.model.config import BATCH_SIZE

# load dataset
df = pd.read_csv("./transformer/data/eng_-french.csv")

# seperate languages
english_df = df['English words/sentences']
french_df = df['French words/sentences']

english_id_seqs = create_id_seqs(english_df)
french_id_seqs = create_id_seqs(french_df)

dataset = TranslationDataset(english_id_seqs, french_id_seqs)
dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

# 1. Convert the dataloader into an iterator
data_iter = iter(dataloader)

# 2. Grab the very first batch
src_batch, trg_batch = next(data_iter)

# 3. Inspect your data shapes
#print("Source batch shape:", src_batch.shape)
#print("Target batch shape:", trg_batch.shape)


