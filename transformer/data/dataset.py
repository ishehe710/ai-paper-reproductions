import pandas as pd
from transformer.data.tokenizer import tokenize
from transformer.data.vocabulary import map_token_to_id, make_ids
import torch
from torch.utils.data import Dataset
from transformer.model.config import FRENCH, ENGLISH


class TranslationDataset(Dataset):

    def __init__(self, input, output):
        """1.

        Initialize data, labels, and transforms here.
        """
        self.input_lang = input
        self.output_lang = output

    def __len__(self):
        """2.

        Return the total number of items in the dataset.
        """
        return len(self.input_lang)

    def __getitem__(self, idx):
        """3.

        Fetch one sample and its label at the given index.
        """
        lang_sample = self.input_lang[idx]
        lang_label = self.output_lang[idx]

        # Convert to PyTorch tensors before returning
        return torch.tensor(lang_sample), torch.tensor(lang_label)


# helper functions
def longest_sequence(lang_tokens: list[list[str]]) -> int:
    longest = 0
    
    for tokens in lang_tokens:
        length = len(tokens)
        if length > longest:
            longest = length
    
    return longest

def add_special_tokens(lang_tokens: list[list[str]]):
    
    for tokens in lang_tokens:
        tokens.insert(0, '<SOS>')
        tokens.append('<EOS>')

def add_padding(lang_tokens: list[list[str]], vector_length: int):
    
    for tokens in lang_tokens:
        while len(tokens) < vector_length:
            tokens.append('<PAD>')
            
def create_id_seqs(lang_df) -> list[list[int]]:
    
    # tokenize language
    lang_tokens = tokenize(lang_df)
    
    # vocab mapping
    lang_mapping = map_token_to_id(lang_tokens)
    
    # create tokens ids
    add_special_tokens(lang_tokens)
    lang_longest = longest_sequence(lang_tokens)
    add_padding(lang_tokens, lang_longest)
    lang_id_seqs = make_ids(lang_tokens, lang_mapping)
    
    return lang_id_seqs
    
def create_dataset(filename: str) -> Dataset:
    
    # load dataset
    df = pd.read_csv(filename)

    # seperate languages
    english_df = df[ENGLISH]
    french_df = df[FRENCH]

    english_id_seqs = create_id_seqs(english_df)
    french_id_seqs = create_id_seqs(french_df)

    dataset = TranslationDataset(english_id_seqs, french_id_seqs)
    
    return dataset

def find_vocab_size():
    pass




        

