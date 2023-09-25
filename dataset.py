from torch.utils.data import random_split, TensorDataset
import pandas as pd
import numpy as np
import glob
import random
import torch

def seq_encoder(seq, max_len, unique_char):
    """
    Function defined using all unique characters in our
    processed structures created
    with the preprocessing_data function.

    Parameters
    ----------
    seq : str
         sequence in string format.
    unique_char : list
         List of unique characters in the string data set.
    max_len : int
         Maximum length of the sequence.

    Returns
    -------
    sequence_matrix : numpy.ndarray
         One-hot encoded matrix of fixed shape
         (unique char in sequence, max sequence length).
    """
    # create dictionary of the unique char data set
    seq2index = {char: index for index, char in enumerate(unique_char)}
    # one-hot encoding
    # zero padding to max_len
    seq_matrix = np.zeros((max_len,len(unique_char)))
    for index in range(min(len(seq),max_len)):
        char = seq[index]
        seq_matrix[index,seq2index[char]] = 1
    return seq_matrix

def pro_encoder(df,type):
    if type == 'origin':
        select = 'Fragment'
    elif type == 'mutant':
        select = 'Sequence'
    else:
        None
    pro_seq = df[select].tolist()
    unique_pro = list('GAVLIPFYWSTCMNQDEKRH'); max_pro = 1000
    pro_encode = [seq_encoder(i,max_pro,unique_pro) for i in pro_seq]
    pro_encode = np.array(pro_encode)
    return pro_encode

def na_encoder(df):
    na_seq = []
    row = df.shape[0]
    for i in range(row):
        na_seq.append(df.iloc[i,14])
    unique_na = list('agcu'); max_na = 100
    na_encode = [seq_encoder(i, max_na, unique_na) for i in na_seq]
    na_encode = np.array(na_encode)
    return na_encode

def hhblits_encoder(hhm_path,df):
    mut_hhb = []
    ori_hhb = []
    seq_len = 1000
    row = df.shape[0]

    for i in range(row):
        # retrieve mut hhb
        mut_path = hhm_path + '/pro_mut_' + str(df.iloc[i,0]) + '.hhm'
        ori_path = hhm_path + '/pro_ori_' + str(df.iloc[i,0]) + '.hhm'
        mut_matrix = np.loadtxt(mut_path)
        ori_matrix = np.loadtxt(ori_path)

        # padding
        if mut_matrix.shape[0]<seq_len:
            z = np.zeros((seq_len-mut_matrix.shape[0],30))
            mut_matrix = np.vstack((mut_matrix,z))
        else:
            mut_matrix = mut_matrix[:seq_len,:]
        
        if ori_matrix.shape[0]<seq_len:
            z = np.zeros((seq_len-ori_matrix.shape[0],30))
            ori_matrix = np.vstack((ori_matrix,z))
        else:
            ori_matrix = ori_matrix[:seq_len,:]
        
        mut_hhb.append(mut_matrix)
        ori_hhb.append(ori_matrix)
    mut_hhb = np.array(mut_hhb)
    ori_hhb = np.array(ori_hhb)
    return ori_hhb, mut_hhb

# def train_val_test_split(df, pro_ori, pro_mut ,hhb_ori ,hhb_mut ,na):
#     row = df.shape[0]
#     full_list = list(range(row))
#     random.shuffle(full_list)

#     idx_tr = round(row*0.8);idx_ts = round(row*0.9)
#     sam_train = full_list[:idx_tr]
#     sam_val = full_list[idx_tr:idx_ts]
#     sam_test = full_list[idx_ts:]

#     y = df['ddG(kcal/mol)'].values.tolist()

#     XY_train = [[pro_ori[i] for i in sam_train],
#                [pro_mut[i] for i in sam_train],
#                [hhb_ori[i] for i in sam_train],
#                [hhb_mut[i] for i in sam_train],
#                [na[i] for i in sam_train],
#                [y[i] for i in sam_train]
#                ]
#     XY_val = [[pro_ori[i] for i in sam_val],
#                [pro_mut[i] for i in sam_val],
#                [hhb_ori[i] for i in sam_val],
#                [hhb_mut[i] for i in sam_val],
#                [na[i] for i in sam_val],
#                [y[i] for i in sam_val]
#                ]
#     XY_test = [[pro_ori[i] for i in sam_test],
#                [pro_mut[i] for i in sam_test],
#                [hhb_ori[i] for i in sam_test],
#                [hhb_mut[i] for i in sam_test],
#                [na[i] for i in sam_test],
#                [y[i] for i in sam_test]
#                ]
#     return XY_train, XY_val, XY_test


class tensorDataset(object):

    def __init__(self, root_dir, data_csv, hhblits_dir):
        self.root_dir = root_dir
        self.data_csv = root_dir + data_csv
        self.hhblits_dir = root_dir + hhblits_dir
        self.df = pd.read_csv(self.data_csv)
    
    def data(self):
        pro_ori = pro_encoder(self.df,'origin'); pro_ori = torch.tensor(pro_ori, dtype=torch.float)
        pro_mut = pro_encoder(self.df,'mutant'); pro_mut = torch.tensor(pro_mut, dtype=torch.float)
        na = na_encoder(self.df); na = torch.tensor(na, dtype=torch.float)
        hhb_ori, hhb_mut = hhblits_encoder(self.hhblits_dir,self.df)
        hhb_ori, hhb_mut = torch.tensor(hhb_ori, dtype=torch.float), torch.tensor(hhb_mut, dtype=torch.float)
        y = self.df['ddG(kcal/mol)'].values.tolist(); y = torch.tensor(y, dtype=torch.float)

        row = self.df.shape[0]
        idx_tr = round(row*0.8);idx_ts = round(row*0.1)
        ran_spl = [idx_tr,idx_ts,row-idx_tr-idx_ts]

        dataset = TensorDataset(pro_ori, pro_mut, hhb_ori, hhb_mut, na, y)
        train_dataset, val_dataset, test_dataset = random_split(dataset, lengths = ran_spl)

        return train_dataset, val_dataset, test_dataset

def pdb_encoder(pdb_path,df):
    ori_hhb = []
    seq_len = 1000
    row = df.shape[0]

    for i in range(row):
        # retrieve mut hhb
        
        ori_path = pdb_path + '/pro_ori_' + str(df.iloc[i,0]) + '.txt'
        ori_matrix = np.loadtxt(ori_path)

        # padding
        if ori_matrix.shape[0]<seq_len:
            z = np.zeros((seq_len-ori_matrix.shape[0],30))
            ori_matrix = np.vstack((ori_matrix,z))
        else:
            ori_matrix = ori_matrix[:seq_len,:]
        
        ori_hhb.append(ori_matrix)
    ori_hhb = np.array(ori_hhb)
    return ori_hhb
