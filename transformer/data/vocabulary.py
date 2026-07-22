
def count_frequencies(lang_tokens: list[list[str]]):
    
    frequencies = {}
    
    for tokens in lang_tokens:
        for token in tokens:
            if token not in frequencies:
                frequencies[token] = 1
            else:
                frequencies[token] += 1
    
    return frequencies
    
    

# create the mapping
def map_token_to_id(lang_tokens: list[list[str]]):
    
    id = 4

    token_to_id = {
        '<PAD>': 0, 
        '<SOS>': 1, # start of sequence
        '<EOS>': 2, # end of sequence
        '<UNK>': 3  # unknown word not correctly found in vocabulary
    }

    
    for tokens in lang_tokens:
        for token in tokens:
            if token not in token_to_id:
                token_to_id[token] = id
                id += 1
    
    return token_to_id

# inverse mapping
def map_id_to_token(mapping: dict):
    
    inv_map = {}
    
    for key, value in mapping.items():
        inv_map[value] = key
        
    return inv_map


# perform mapping: can be either token to id, or id to token, via parameter
def do_map(token_seq: list[str], vocab):
    
    id_seq = []
    for token in token_seq:
        id_seq.append(vocab[token])
    
    return id_seq
        
def make_ids(lang_tokens: list[list[str]], token_to_id_map) -> list[list[int]]:
    
    # id sequences of the language
    id_seqs = []
    
    for tokens in lang_tokens:
        
        id_seqs.append(do_map(tokens, token_to_id_map))
    
    return id_seqs