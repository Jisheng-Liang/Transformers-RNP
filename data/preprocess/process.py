import pandas as pd

file = 'lang_rna.csv'
df = pd.read_csv(file)

row,col =df.shape

for i in range(row):
    seq = df.iloc[i,4]
    seq = seq.replace('\n','')
    seq = seq.upper()
    df.iloc[i,4] = seq

# protein for direct prediction
df.to_csv('0.csv',index=False)

# mutant protein and unmutant RNA
file = '0.csv'
df = pd.read_csv(file)
df = df[df['Type_nuc'] == 'RNA']
row,col = df.shape
tab = 'AUGCaugc'
tab = list(tab)
number = list('0123456789')
for j in range(row):
    seq = list(df.iloc[j,14])
    new_seq = ""
    for k in range(len(seq)):
        if seq[k] in tab:
            new_seq += seq[k]
        else:
            if seq[k] == "(":
                base = ""
                k += 1
                while(k < len(seq) and seq[k] != ")"):
                    if seq[k] in tab:
                        base += seq[k]
                    k += 1
                k += 1
                num = ""
                while(k < len(seq) and seq[k] in number):
                    num += seq[k]
                    k += 1
                if num == "": num = 1
                new_seq += base * int(num)
    new_seq = new_seq.lower()
    if len(new_seq) < 4:
        new_seq = '-'
    df.iloc[j,14] = new_seq

df = df[df['Sequence_wild1'] != '-']

print(df.shape)
df.to_csv('1_rna.csv', index=False)

# # mutant protein and mutant NA
# row,col = df.shape
# tab = 'ATGCUatgcu '
# tab = list(tab)
# print(row)
# for j in range(row):
#     seq = list(df.iloc[j,17])
#     new_seq = []
#     exit_flag = False
#     for k in range(len(seq)):
#         if seq[k] in tab:
#             new_seq.append(seq[k])
#         else:
#             df.iloc[j,17] = '-'
#             exit_flag = True
#             break
#     if exit_flag:
#         continue
#     new_seq = ''.join(new_seq)
#     new_seq = new_seq.lower()
#     df.iloc[j,17] = new_seq
# df = df[df['Sequence_mutant_2'] != '-']
# print(df.shape)
# df.to_csv('1_mutNA.csv', index=False)

# code for mutant Protein
file = '1_rna.csv'
df = pd.read_csv(file)
row, col = df.shape
for i in range(row):
    seq = df.iloc[i,10]
    # unmutant protein unchanged
    if seq == 'wild':
        continue
    pro_s = list(df.iloc[i,4])
    # print(seq)
    if ',' in seq:
        seq = seq.split(', ')
        print(seq)
        for j in range(len(seq)):
            if seq[j] == "?CT":
                continue
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

Tab = 'GAVLIPFYWSTCMNQDEKRH'
Tab = list(Tab)
row, col = df.shape
for i in [2,4]:
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
df = df[df['ddG(kcal/mol)'] != '-']
df = df[df['Fragment'] != '-']

df = df.drop_duplicates()
df.to_csv('ddg.csv',index=False)
print(df.shape)
