import pandas as pd
import torch
from transformer.data.dataset import create_id_seqs, TranslationDataset
from torch.utils.data import Dataset, DataLoader
from transformer.model.config import BATCH_SIZE



def create_dataloader(dataset, batch_size, shuffle=False, generator=None) -> DataLoader:

    
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=generator)

    return dataloader
