import pandas as pd
import torch
from transformer.data.dataset import create_id_seqs, TranslationDataset
from torch.utils.data import Dataset, DataLoader
from transformer.model.config import BATCH_SIZE



def create_dataloader(dataset, batch_size, shuffle=False) -> DataLoader:

    
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader
