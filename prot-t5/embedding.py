from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
import pandas as pd
import numpy as np

def main():
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # prepare your protein sequences as a list
    df = pd.read_csv('data_1/final_rna.csv')
    print(df.shape)
    sequence_examples = df['Sequence'].T.values.tolist()

    # replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

    id_examples = df['Entry_id'].T.values.tolist()

    # Load the tokenizer
    tokenizer = T5Tokenizer.from_pretrained('/data/personal/liangjs/ProtT5-XL-UniRef50/', do_lower_case=False)

    # Load the model
    model = T5EncoderModel.from_pretrained('/data/personal/liangjs/ProtT5-XL-UniRef50/').to(device)

    # only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
    model.full() if device=='cpu' else model.half()

    # tokenize sequences and pad up to the longest sequence in the batch
    fp = np.memmap("/data/personal/liangjs/prot_emb1_ori.dat", dtype='float32', mode='w+', shape=(df.shape[0],1000,1024))
    for i in range(df.shape[0]):
        ids = tokenizer([sequence_examples[i]], add_special_tokens=True, truncation=True, padding='max_length', max_length=1000)

        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        # generate embeddings
        with torch.no_grad():
            embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

        # # extract residue embeddings for the first ([0,:]) sequence in the batch and remove padded & special tokens ([0,:7]) 
        # emb_0 = embedding_repr.last_hidden_state[0,:7] # shape (7 x 1024)
        # # same for the second ([1,:]) sequence but taking into account different sequence lengths ([1,:8])
        # emb_1 = embedding_repr.last_hidden_state[1,:8] # shape (8 x 1024)
        fp[i,:] = embedding_repr.last_hidden_state[0,:1000].cpu().numpy()
        print(i)
    # if you want to derive a single representation (per-protein embedding) for the whole protein
        # emb_protein = embedding_repr.last_hidden_state[0,:1000].mean(dim=0) # shape (1024)

if __name__ == "__main__":
    main()