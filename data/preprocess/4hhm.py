import pandas as pd
import os

df = pd.read_csv('ddg1.csv')
for i in range(df.shape[0]):
    id = df.iloc[i,0]
    ori_path = '/data/personal/liangjs/matrix/pro_ori_' + str(id) + '.hhm'
    if os.path.exists(ori_path):
        continue
    else:
        with open(('../4hhm/pro_ori_'+str(id)+'.fasta'), 'w+') as f:
            f.write('>'+str(id)+'\n')
            f.write(df.iloc[i,4])
    ori_path = '/data/personal/liangjs/matrix/pro_mut_' + str(id) + '.hhm'
    if os.path.exists(ori_path):
        continue
        with open(('../4hhm/pro_mut_'+str(id)+'.fasta'), 'w+') as f:
            f.write('>'+str(id)+'\n')
            f.write(df.iloc[i,2])
        