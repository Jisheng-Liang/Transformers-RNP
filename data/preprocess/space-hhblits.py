# generate mean space-hhblits
import numpy as np
import os
import pandas as pd

df = pd.read_csv("../train.csv")
name_list = [df.iloc[i,0] for i in range(df.shape[0])]
uniprot_list = [df.iloc[i,5].strip(' ') for i in range(df.shape[0])]

max_range = 11
hhm_path = '../hhblits/'
pdb_path = '../pdb/'
output_path = '../space_hhblits/'
for i in range(len(name_list)):
    print(i)
    # fetch length
    mut_path = hhm_path + "pro_ori_" + str(name_list[i]) + '.hhm'
    hhm_matrix = np.loadtxt(mut_path)
    hhm_seq_len = min(hhm_matrix.shape[0], 1000)

    #print(hhm_matrix.shape) # # seq_len*30
    #print(hhm_matrix)
    
    # fetch Ca 3d-coord from .pdb
    uniprot_id = uniprot_list[i]
    if uniprot_id == '-':
        uniprot_id = 'S2N3J9'
    with open(pdb_path + uniprot_id + '.pdb') as pdb_file:
        pdb_matrix = np.zeros([hhm_seq_len, 3], float)
        pdb_line = pdb_file.readline()
        while(pdb_line[0:4] != 'ATOM'):
            pdb_line = pdb_file.readline()
        iddx = 0
        while pdb_line:
            if(pdb_line[0:4] != 'ATOM'):
                break    
            number = pdb_line[22:27].strip()
            CA = pdb_line[13:15]
            if(int(number) == iddx + 1 and CA == 'CA'):
                pdb_matrix[iddx,0] = float(pdb_line[31:38].replace(" ", "")) # x cood
                pdb_matrix[iddx,1] = float(pdb_line[39:46].replace(" ", "")) # y cood
                pdb_matrix[iddx,2] = float(pdb_line[47:54].replace(" ", "")) # z cood
                iddx += 1
                if(iddx >= hhm_seq_len):
                    break
                
            pdb_line = pdb_file.readline()
        #print(pdb_matrix.shape) # seq_len*3
        #print(pdb_matrix)
        
    # spatial filtering
    space_hhm_matrix = np.zeros([hhm_seq_len, 30], float)
    res_dict = {}
    for residue_num in range(hhm_seq_len):
        res_dict[residue_num] = []
        x, y, z = pdb_matrix[residue_num,0],pdb_matrix[residue_num,1],pdb_matrix[residue_num,2]
        for pair in range(hhm_seq_len):
            x_pair, y_pair, z_pair = pdb_matrix[pair,0],pdb_matrix[pair,1],pdb_matrix[pair,2]
            if((x-x_pair)*(x-x_pair) + (y-y_pair)*(y-y_pair) + (z-z_pair)*(z-z_pair) <= max_range*max_range):
                res_dict[residue_num].append(pair)
    for residue_num in range(hhm_seq_len):
        residue_list = res_dict[residue_num]
        for j in range(30):
            for num in residue_list:
                space_hhm_matrix[residue_num][j] += hhm_matrix[num][j]
            space_hhm_matrix[residue_num][j] /= float(len(residue_list)) # average
    #print(space_hhm_matrix.shape)
    #print(space_hhm_matrix)
    
    with open(os.path.join(output_path, 'pro_ori_'+str(name_list[i])+'.shhm'),'w+') as out_file:
        np.savetxt(out_file, space_hhm_matrix, fmt='%.6f')            