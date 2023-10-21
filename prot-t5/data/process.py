import pandas as pd

# file = 'prott5.csv'
# df = pd.read_csv(file,encoding='gb18030')

# # row = df.shape[0]
# # print(row)
# # for i in range(row):
# #     seq = df.iloc[i,4]
# #     seq = seq.replace('\n','')
# #     seq = seq.upper()
# #     df.iloc[i,4] = seq
# # df.to_csv(file, index=False)

# row, col = df.shape
# tab = 'AUGCaugc- '
# tab = list(tab)
# for j in range(row):
#     seq = list(df.iloc[j,14])
#     new_seq = []
#     exit_flag = False
#     for k in range(len(seq)):
#         if seq[k] in tab:
#             if seq[k] == ' ':
#                 continue
#             else:
#                 new_seq.append(seq[k])
#         else:
#             df.iloc[j,14] = '-'
#             exit_flag = True
#             break
#     if exit_flag:
#         continue
#     new_seq = ''.join(new_seq)
#     new_seq = new_seq.upper()
#     df.iloc[j,14] = new_seq
# df = df[df['Sequence_wild1'] != '-']

# row, col = df.shape
# for i in range(row):
#     # mutant rna sequence
#     if df.iloc[i,16] != '-' and df.iloc[i,17] == '-':
#         df.iloc[i,17] = df.iloc[i,16]
#     df.iloc[i,17] = df.iloc[i,17].upper()
#     # label
#     if df.iloc[i,33] != '-' and df.iloc[i,32] == '-':
#         df.iloc[i,32] = df.iloc[i,33]

# df.to_csv('0.csv', index=False)

# code for mutant Protein
file = '0.csv'
df = pd.read_csv(file)
row, col = df.shape

for i in range(row):
    seq = df.iloc[i,10]
    pro_s = list(df.iloc[i,4])
    # print(seq)
    if seq == 'wild':
        None
    elif ',' in seq:
        seq = seq.split(', ')
        print(seq)
        for j in range(len(seq)):
            cha = list(seq[j])
            pos = ''.join(cha[1:len(cha)-1])
            pro_s[int(pos)-1] = cha[len(cha)-1]
    elif 'Del' in seq:
        seq = seq.replace('Del ','')
        if ';' in seq:
            seq.rstrip(';')
            seq = seq.split(';')
            seq = seq[:(len(seq)-1)]
            # print(seq)
            for k in range(len(seq)):
                if '-' in seq[k]:
                    new_list = seq[k].split('-')
                    sta = list(new_list[0])
                    sta = ''.join(sta[1:len(sta)])
                    end = list(new_list[1])
                    end = ''.join(end[1:len(end)])
                    print(sta,end)
                    for l in range(int(sta)-1,int(end)):
                        pro_s[l] = ''
                else:
                    new_list = list(seq[k])
                    if new_list[-1].isalpha():
                        pos = ''.join(new_list[1:len(new_list)-1])
                        pro_s[int(pos)-1] = new_list[len(new_list)-1]
                    else:
                        pos = ''.join(new_list[1:len(new_list)])
                        print(pos, len(pro_s))
                        pro_s[int(pos)-1] = ''
        else:
            if '-' in seq:
                new_list = seq.split('-')
                sta = list(new_list[0])
                sta = ''.join(sta[1:len(sta)])
                end = list(new_list[1])
                end = ''.join(end[1:len(end)])
                # print(sta,end-1)
                for l in range(int(sta)-1,int(end)):
                    pro_s[l] = ''
            else:
                new_list = list(seq)
                pos = ''.join(new_list[1:len(new_list)])
                # print(new_list)
                pro_s[int(pos)-1] = ''
    else:
        cha = list(seq)
        pos = ''.join(cha[1:len(cha)-1])
        # print(i)
        pro_s[int(pos)-1] = cha[len(cha)-1]
    pro_s = ''.join(pro_s)
    df.iloc[i,2] = pro_s.upper()

Tab = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
Tab = list(Tab)
row, col = df.shape
for i in [2]:
    for j in range(row):
        seq = list(df.iloc[j,i])
        new_seq = []
        exit_flag = False
        for k in range(len(seq)):
            if seq[k] in Tab:
                new_seq.append(seq[k])
            else:
                df.iloc[j,i] = '-'
                exit_flag = True
                break
        if exit_flag:
            continue
        new_seq = ''.join(new_seq)
        new_seq = new_seq.upper()
        df.iloc[j,i] = new_seq
# df = df[df['Sequence'] != '-']
df = df[df['Fragment'] != '-']

df = df.drop_duplicates()
df.to_csv('final_rna.csv',index=False)
print(df.shape)
