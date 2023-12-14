import pandas as pd

df = pd.read_csv('ddg1.csv')
df = df.drop(df.columns[[18,19,20,21,22,23,24,25,26,27,37,38,39,40]], axis=1)
df = df.sample(frac=1)

frac = int(df.shape[0] * 0.1)
test = df.iloc[:frac, :]
val = df.iloc[frac:2*frac, :]
train = df.iloc[2*frac:, :]
test.to_csv("test.csv", index=False)
val.to_csv("val.csv", index=False)
train.to_csv("train.csv", index=False)