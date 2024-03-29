import numpy as np
import re
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

def pro_encoder(seq):
    seq = ["".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in seq]
    unique_pro = list('GAVLIPFYWSTCMNQDEKRH'); max_pro = 1000
    pro_encode = [seq_encoder(i,max_pro,unique_pro) for i in seq]
    pro_encode = np.array(pro_encode, dtype=np.float32)
    return pro_encode

def na_encoder(seq):
    na_seq = ["".join(list(re.sub(r"[u]", "t", sequence))) for sequence in seq]
    unique_na = list('agct'); max_na = 100
    na_encode = [seq_encoder(i, max_na, unique_na) for i in na_seq]
    na_encode = np.array(na_encode, dtype=np.float32)
    return na_encode

def hhblits_encoder(hhm_path, id):
    mut_hhb = []
    ori_hhb = []
    seq_len = 1000

    # retrieve mut hhb

    for i in id:
        mut_path = hhm_path + 'pro_mut_' + str(i) + '.hhm'
        ori_path = hhm_path + 'pro_ori_' + str(i) + '.hhm'
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
    mut_hhb = np.array(mut_hhb, dtype=np.float32)
    ori_hhb = np.array(ori_hhb, dtype=np.float32)
    return ori_hhb, mut_hhb

def shhblits_encoder(hhm_path, id):
    mut_hhb = []
    ori_hhb = []
    seq_len = 1000

    # retrieve mut hhb

    for i in id:
        mut_path = hhm_path + 'pro_mut_' + str(i) + '.shhm'
        ori_path = hhm_path + 'pro_ori_' + str(i) + '.shhm'
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
    mut_hhb = np.array(mut_hhb, dtype=np.float32)
    ori_hhb = np.array(ori_hhb, dtype=np.float32)
    return ori_hhb, mut_hhb

def shhblits(ori_path, mut_path):
    mut_hhb = []
    ori_hhb = []
    seq_len = 1000

    # retrieve mut hhb

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

    mut_hhb = np.array(mut_hhb, dtype=np.float32)
    ori_hhb = np.array(ori_hhb, dtype=np.float32)
    return ori_hhb, mut_hhb

def pdb_encoder(df):
    pdblist = []
    row = df.shape[0]
    seq_len = 1000
    pdb_root = "data/pdb/"
    for i in range(row):
        UniProt_ID = df.iloc[i,5]
        pdb_path = pdb_root + UniProt_ID + '.txt'
        pdb_matrix = np.loadtxt(pdb_path)

        # padding
        if pdb_matrix.shape[0]<seq_len:
            z = np.zeros((seq_len-pdb_matrix.shape[0],4))
            pdb_matrix = np.vstack((pdb_matrix,z))
        else:
            pdb_matrix = pdb_matrix[:seq_len,:]
        
        pdblist.append(pdb_matrix)
    pdb_encode = np.array(pdblist)
    return pdb_encode

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
