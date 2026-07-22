import pandas as pd

def tokenize(lang_col: pd.DataFrame) -> list[str]:
    
    # extract values from dataframe
    sentences: list[str] = lang_col.values.tolist()
    
    # tokenizing sentences
    tokens = []
    for sentence in sentences:
        tokens.append(sentence.split(" "))
    
    return tokens
    