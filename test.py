import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from dataset import pro_encoder, na_encoder, hhblits_encoder
from model import Trans
import pandas as pd
import math

def main():
    batch_size = 128
    root_dir = "./"
    data_csv = "final_rna.csv"
    hhblits_dir = "hhblits/"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = []
    for i in ['train','val','test']:
        filename = i+'.csv'
        df = pd.read_csv(filename)
        pro_ori = pro_encoder(df,'origin'); pro_ori = torch.tensor(pro_ori, dtype=torch.float)
        pro_mut = pro_encoder(df,'mutant'); pro_mut = torch.tensor(pro_mut, dtype=torch.float)
        na = na_encoder(df); na = torch.tensor(na, dtype=torch.float)
        hhb_ori, hhb_mut = hhblits_encoder(hhblits_dir,df)
        hhb_ori, hhb_mut = torch.tensor(hhb_ori, dtype=torch.float), torch.tensor(hhb_mut, dtype=torch.float)
        y = df['ddG(kcal/mol)'].values.tolist(); y = torch.tensor(y, dtype=torch.float)
        dataset.append(TensorDataset(pro_ori, pro_mut, hhb_ori, hhb_mut, na, y))

    train_dataset = dataset[0]; val_dataset = dataset[1]; test_dataset = dataset[2]
    print(len(train_dataset),len(val_dataset), len(test_dataset))

    loss_fn = nn.MSELoss()
    model = Trans().to(device)
    # test
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

    test_mse = 0.
    test_result = []
    model.load_state_dict(torch.load('result/trans_best_all.pth'))
    model.eval()
    for X_ori, X_mut, X_orihhb, X_muthhb, X_na, y in test_loader:
        X_mut = X_mut.to(device); X_muthhb = X_muthhb.to(device)
        X_ori = X_ori.to(device); X_orihhb = X_orihhb.to(device)
        X_na = X_na.to(device)
        y = y.to(device)
        preds = model(X_ori, X_mut, X_orihhb, X_muthhb, X_na)
        loss = loss_fn(preds.ravel(), y)
        
        test_result.append(y.tolist())
        test_result.append(preds.tolist())
        test_result.append(loss.tolist())
        test_mse += loss.item()
    # test_mse = test_mse/len(test_dataset)
    test_rmse = math.sqrt(test_mse)

    file_name = 'result/trans_test_result.txt'
    f = open(file_name,'w')
    for i in test_result:
        f.write(str(i)+'\n')
    f.close()
    print(test_rmse)

if __name__ == "__main__":
    main()
