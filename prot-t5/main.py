from model import prott5, prot_only
import torch
import torch.nn as nn
from torch.optim import Adam
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import math, os, tqdm

def main():
    
    epochs = 200
    batch_size = 64
    pro_dir = '/data/personal/liangjs/prot_emb1.dat'
    ori_dir = '/data/personal/liangjs/prot_emb1_ori.dat'
    rna_dir = '/data/personal/liangjs/rna_emb1.dat'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pro_emb = np.memmap(pro_dir, dtype='float32', mode='r', shape=(1125,1000,1024))
    ori_emb = np.memmap(ori_dir, dtype='float32', mode='r', shape=(1125,1000,1024))
    rna_emb = np.memmap(rna_dir, dtype='float32', mode='r', shape=(1125,102,640))

    df = pd.read_csv('./data_1/final_rna.csv')
    for i in range(df.shape[0]):
        df.iloc[i,3] = i
    df.to_csv('./data_1/final_rna.csv', index=False)
    
    df = df.sample(frac=1, random_state=40)
    train = df.iloc[:int(df.shape[0]*0.8),:]; train.to_csv('./data_1/train.csv',index=False)
    val = df.iloc[int(df.shape[0]*0.8):int(df.shape[0]*0.9),:]; val.to_csv('./data_1/val.csv',index=False)
    test = df.iloc[int(df.shape[0]*0.9):,:]; test.to_csv('./data_1/test.csv',index=False)

    # prepare your protein sequences as a list
    dataset = []
    # df = pd.read_csv('./data_1/final_rna.csv')
    for i in ['train','val','test']:

        filename = "./data_1/"+i+'.csv'
        data = pd.read_csv(filename)
        for j in range(data.shape[0]):
            candidate = data.iloc[j,0]
            index = df[df.Entry_id == candidate].index.tolist()
            data.iloc[j,3] = index[0]
        
        # filename = "./data_1/"+i+'.csv'
        # df = pd.read_csv(filename)
        index = data['Protein_Source'].T.values.tolist()
        pro, ori, rna = [], [], []
        for i in range(len(index)):
            pro.append(pro_emb[index[i]])
            ori.append(ori_emb[index[i]])
            rna.append(rna_emb[index[i]])
        y = data['ddG(kcal/mol)'].T.values.tolist()
        pro = torch.tensor(np.array(pro), dtype=torch.float)
        ori = torch.tensor(np.array(ori), dtype=torch.float)
        rna = torch.tensor(np.array(rna), dtype=torch.float)

        for i in range(len(y)):
            if y[i] == '-':
                y[i] = 0
                continue
            y[i] = float(y[i])
        y = torch.tensor(y, dtype=torch.float)
        dataset.append(TensorDataset(pro, ori, rna, y))
    
    train_dataset = dataset[0]; val_dataset = dataset[1]; test_dataset = dataset[2]
    print(len(train_dataset),len(val_dataset), len(test_dataset))
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size)

    loss_fn = nn.MSELoss()
    model = prot_only().to(device)
    print(model)
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: 0.99**epoch)

    # train
    min_val_loss = float("inf")
    log = []
    for epoch in range(1, epochs+1):
        total_loss = 0.
        best_model = None
        model.train()
        for pro, ori, rna, y in train_loader:
            optimizer.zero_grad()
            pro = pro.to(device); ori = ori.to(device); rna = rna.to(device)
            y_preds = model(pro, ori, rna)
            y = y.to(device)
            loss = loss_fn(y_preds.ravel(), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*len(pro)
        total_mse = total_loss/len(train_dataset)

        # validate
        model.eval()
        val_losses = 0.
        with torch.no_grad():
            for pro, ori, rna, y in val_loader:
                pro = pro.to(device); ori = ori.to(device); rna = rna.to(device)
                y_preds = model(pro, ori, rna)
                y = y.to(device)
                val_loss = loss_fn(y_preds.ravel(), y)
                val_losses += val_loss.item()*len(pro)
        val_mse = val_losses/len(val_dataset)

        log.append([epoch,total_mse,val_mse])
        print('| epoch {:3d} | lr {:5.5f} '
                      'train_loss {:5.5f} | val_loss {:5.5f} | val_rmse {:5.5f}'.
                      format(epoch, scheduler.get_last_lr()[0], total_mse, val_mse, math.sqrt(val_mse)))
        if (epoch >= 10) and (val_mse < min_val_loss):
            min_val_loss = val_loss
            best_model = model
            torch.save(best_model.state_dict(), 'result/protonly.pth')
        scheduler.step()
    f = open('result/trans_log_only.txt','w')
    for i in log:
        f.write(str(i)+'\n')
    f.close()

    # test
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

    test_mse = 0.
    test_result = []
    # model = prott5.to(device)
    # model.load_state_dict(torch.load('result/prott5.pth'))
    model.eval()
    for pro, ori, rna, y in test_loader:
        pro = pro.to(device); ori = ori.to(device); rna = rna.to(device)
        y_preds = model(pro, ori, rna)
        y = y.to(device)
        loss = loss_fn(y_preds.ravel(), y)
        
        test_result.append(y.tolist())
        test_result.append(y_preds.tolist())
        test_result.append(loss.tolist())
        test_mse += loss.item()*len(pro)
    test_mse = test_mse/len(test_dataset)
    test_rmse = math.sqrt(test_mse)

    file_name = 'result/protonly.txt'
    f = open(file_name,'w')
    for i in test_result:
        f.write(str(i)+'\n')
    f.close()
    print(test_rmse)

if __name__ == "__main__":
    main()