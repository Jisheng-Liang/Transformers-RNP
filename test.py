import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from dataset import pro_encoder, hhblits_encoder, na_encoder, shhblits_encoder
from model import Trans, MLPRegressor, CNNRegressor, LSTMRegressor, BLSTMRegressor
import pandas as pd
import math, os
from tqdm import tqdm

def main():    
    batch_size = 32
    root_dir = "data/"
    hhblits_dir = root_dir+"hhblits/"
    dataset = []
    for i in ['test']:
        filename = root_dir + i +'.csv'
        df = pd.read_csv(filename)
        id = df.iloc[:,0].values.tolist(); na_seq = df.iloc[:,14].values.tolist()
        mut_seq = df.iloc[:,2].values.tolist(); ori_seq = df.iloc[:,4].values.tolist()
        
        pro_ori = pro_encoder(ori_seq); pro_ori = torch.tensor(pro_ori, dtype=torch.float)
        pro_mut = pro_encoder(mut_seq); pro_mut = torch.tensor(pro_mut, dtype=torch.float)
        na = na_encoder(na_seq); na = torch.tensor(na, dtype=torch.float)
        hhb_ori, hhb_mut = hhblits_encoder(hhblits_dir,id)
        hhb_ori, hhb_mut = torch.tensor(hhb_ori, dtype=torch.float), torch.tensor(hhb_mut, dtype=torch.float)
        
        y = df['ddG(kcal/mol)'].values.tolist(); y = torch.tensor(y, dtype=torch.float)
        dataset.append(TensorDataset(pro_ori, pro_mut, hhb_ori, hhb_mut, na, y))

    test_dataset = dataset[0]
    print(len(test_dataset))
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    loss_fn = nn.MSELoss()
    model = MLPRegressor().cuda()
    # model = torch.nn.DataParallel(model).cuda()

    # test
    test_mse = 0.
    test_result = []
    model.load_state_dict(torch.load('/data/personal/liangjs/result_rna/mlp.pth'))
    model.eval()
    for X_ori, X_mut, X_orihhb, X_muthhb, X_na, y in tqdm(test_loader):
        X_mut = X_mut.cuda(); X_muthhb = X_muthhb.cuda()
        X_ori = X_ori.cuda(); X_orihhb = X_orihhb.cuda()
        X_na = X_na.cuda()
        y = y.cuda()
        preds = model(X_ori, X_mut, X_orihhb, X_muthhb, X_na)
        loss = loss_fn(preds.ravel(), y)
        # test_result.append(y.tolist())
        test_result.append(preds.squeeze(1).tolist())
        test_mse += loss.item()*len(y)
    test_mse = test_mse/len(test_dataset)
    test_rmse = math.sqrt(test_mse)
    test_result.append(test_rmse)

    file_name = 'result/mlp_test_result.txt'
    f = open(file_name,'w')
    for i in test_result:
        f.write(str(i).strip('[').strip(']')+', ')
    f.close()
    print(test_rmse)

if __name__ == "__main__":
    main()
