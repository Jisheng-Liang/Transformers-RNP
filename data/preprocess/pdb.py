import pandas as pd
import os

df = pd.read_csv('ddg.csv')

path = '../pdb'
exist_path = [i.split('-')[1] for i in os.listdir(path)]
print(len(exist_path))
find_path = []
for i in range(df.shape[0]):
    if df.iloc[i,5].strip(' ') not in exist_path:
        find_path.append(i)
df = df.drop(find_path, axis=0)
# df.to_csv('ddg1.csv', index=False)
for i in os.listdir(path):
    after = i.split('-')[1] + '.pdb'
    os.rename(os.path.join(path,i), os.path.join(path, after))