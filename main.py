import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from dataset import pro_encoder, hhblits_encoder, na_encoder, pdb_encoder
from model import Trans_pdb
import pandas as pd
import math, os

def main():
    epochs = 200
    batch_size = 128
    hhblits_dir = "data/hhblits/"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = []
    for i in ['train','val','test']:
        filename = "data/"+i+'.csv'
        df = pd.read_csv(filename)
        pro_ori = pro_encoder(df,'origin'); pro_ori = torch.tensor(pro_ori, dtype=torch.float)
        pro_mut = pro_encoder(df,'mutant'); pro_mut = torch.tensor(pro_mut, dtype=torch.float)
        na = na_encoder(df); na = torch.tensor(na, dtype=torch.float)
        pdb = pdb_encoder(df); pdb = torch.tensor(pdb, dtype=torch.float)
        hhb_ori, hhb_mut = hhblits_encoder(hhblits_dir,df)
        hhb_ori, hhb_mut = torch.tensor(hhb_ori, dtype=torch.float), torch.tensor(hhb_mut, dtype=torch.float)
        y = df['ddG(kcal/mol)'].values.tolist(); y = torch.tensor(y, dtype=torch.float)
        dataset.append(TensorDataset(pro_ori, pro_mut, hhb_ori, hhb_mut, na, pdb, y))

    train_dataset = dataset[0]; val_dataset = dataset[1]; test_dataset = dataset[2]
    print(len(train_dataset),len(val_dataset), len(test_dataset))
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size)

    loss_fn = nn.MSELoss()
    model = Trans_pdb().to(device)

    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: 0.99**epoch)

    # train
    min_val_loss = float("inf")
    log = []
    for epoch in range(1, epochs+1):
        total_loss = 0.
        best_model = None
        model.train()
        for X_ori, X_mut, X_orihhb, X_muthhb, X_na, X_pdb, y in train_loader:
            optimizer.zero_grad()
            X_ori = X_ori.to(device)
            X_mut = X_mut.to(device)
            X_muthhb = X_muthhb.to(device)
            X_orihhb = X_orihhb.to(device)
            X_na = X_na.to(device); X_pdb = X_pdb.to(device)
            y_preds = model(X_ori, X_mut, X_orihhb, X_muthhb, X_na, X_pdb)
            y = y.to(device)
            loss = loss_fn(y_preds.ravel(), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*len(X_ori)
        total_mse = total_loss/len(train_dataset)

        # validate
        model.eval()
        val_losses = 0.
        with torch.no_grad():
            for X_ori, X_mut, X_orihhb, X_muthhb, X_na, X_pdb, y in val_loader:
                X_ori = X_ori.to(device)
                X_mut = X_mut.to(device)
                X_muthhb = X_muthhb.to(device)
                X_orihhb = X_orihhb.to(device)
                X_na = X_na.to(device); X_pdb = X_pdb.to(device)
                y = y.to(device)
                preds = model(X_ori, X_mut, X_orihhb, X_muthhb, X_na, X_pdb)
                val_loss = loss_fn(preds.ravel(), y)
                val_losses += val_loss.item()*len(X_ori)
        val_mse = val_losses/len(val_dataset)

        log.append([epoch,total_mse,val_mse])
        print('| epoch {:3d} | lr {:02.5f} '
                      'train_loss {:5.5f} | val_loss {:5.5f} | val_rmse {:5.5f}'.
                      format(epoch, scheduler.get_last_lr()[0],total_mse, val_mse, math.sqrt(val_mse)))
        if (epoch >= 50) and (val_mse < min_val_loss):
            min_val_loss = val_loss
            best_model = model
            torch.save(best_model.state_dict(), 'result/trans_pdb.pth')
        scheduler.step()
    f = open('result/trans_log.txt','w')
    for i in log:
        f.write(str(i)+'\n')
    f.close()

    # test
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

    test_mse = 0.
    test_result = []
    model = Trans_pdb.to(device)
    model.load_state_dict(torch.load('result/trans_pdb.pth'))
    model.eval()
    for X_ori, X_mut, X_orihhb, X_muthhb, X_na, X_pdb, y in test_loader:
        X_mut = X_mut.to(device); X_muthhb = X_muthhb.to(device)
        X_ori = X_ori.to(device); X_orihhb = X_orihhb.to(device)
        X_na = X_na.to(device); X_pdb = X_pdb.to(device)
        y = y.to(device)
        preds = model(X_ori, X_mut, X_orihhb, X_muthhb, X_na, X_pdb)
        loss = loss_fn(preds.ravel(), y)
        
        test_result.append(y.tolist())
        test_result.append(preds.tolist())
        test_result.append(loss.tolist())
        test_mse += loss.item()*len(X_ori)
    test_mse = test_mse/len(test_dataset)
    test_rmse = math.sqrt(test_mse)

    file_name = 'result/trans_pdb.txt'
    f = open(file_name,'w')
    for i in test_result:
        f.write(str(i)+'\n')
    f.close()
    print(test_rmse)

if __name__ == "__main__":
    main()
