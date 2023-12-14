import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import pro_encoder, hhblits_encoder, na_encoder, shhblits_encoder
from model import Trans, MLPRegressor, CNNRegressor, LSTMRegressor, BLSTMRegressor
import pandas as pd
import math

from tqdm import tqdm

from torch.optim import Adam

def main():

    epochs = 200
    batch_size = 32
    root_dir = "data/"
    hhblits_dir = root_dir + "hhblits/" 

    dataset = []
    for i in ['train','val','test']:
        filename = root_dir+i+'.csv'
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

    train_dataset = dataset[0]; val_dataset = dataset[1]; test_dataset = dataset[2]
    print(len(train_dataset),len(val_dataset), len(test_dataset))
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, pin_memory=True)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, pin_memory=True)
    loss_fn = nn.MSELoss()
    model = MLPRegressor().cuda()
    # model = torch.nn.DataParallel(model).cuda()

    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[202], gamma=0.1)

    # train
    best_val_loss = float("inf")
    log = []
    for epoch in range(1, epochs+1):
        # train
        total_loss = 0.
        model.train()
        for X_ori, X_mut, X_orihhb, X_muthhb, X_na, y in tqdm(train_loader):
            optimizer.zero_grad()
            X_ori = X_ori.cuda()
            X_mut = X_mut.cuda()
            X_muthhb = X_muthhb.cuda()
            X_orihhb = X_orihhb.cuda()
            X_na = X_na.cuda()
            y_preds = model(X_ori, X_mut, X_orihhb, X_muthhb, X_na)
            y = y.cuda()

            loss = loss_fn(y_preds.ravel(), y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()*len(X_ori)
        total_loss = torch.tensor(total_loss).cuda()
        total_mse = total_loss/len(train_dataset)
        # validate
        model.eval()
        val_losses = 0.
        with torch.no_grad():
            for X_ori, X_mut, X_orihhb, X_muthhb, X_na, y in tqdm(val_loader):
                X_ori = X_ori.cuda()
                X_mut = X_mut.cuda()
                X_muthhb = X_muthhb.cuda()
                X_orihhb = X_orihhb.cuda()
                X_na = X_na.cuda()
                y = y.cuda()
                preds = model(X_ori, X_mut, X_orihhb, X_muthhb, X_na)
                val_loss = loss_fn(preds.ravel(), y)
                val_losses += val_loss.item()*len(X_ori)
        val_losses = torch.tensor(val_losses).cuda()
        val_mse = val_losses/len(val_dataset)
        

        log.append([epoch,total_mse.item(),val_mse.item()])
        print('| epoch {:3d} | lr {:02.5f} '
                      'train_loss {:5.5f} | val_loss {:5.5f} | val_rmse {:5.5f}'.
                      format(epoch, scheduler.get_last_lr()[0],total_mse, val_mse, math.sqrt(val_mse)))
        if val_mse < best_val_loss and epoch > 30:
            best_val_loss = val_mse
            best_model = model
            torch.save(best_model.state_dict(), 'result_rna/mlp.pth')
        scheduler.step()
    file_name = 'result_rna/mlplog_' + '.txt'   
    f = open(file_name,'w')
    for i in log:
        f.write(str(i).strip('[').strip(']')+'\n')
    f.close()

    # # test
    # dist.barrier()
    # test_result = []
    # test_mse = 0.
    # model.eval()
    # for X_ori, X_mut, X_orihhb, X_muthhb, X_na, y in tqdm(test_loader):
    #     X_mut = X_mut.cuda(); X_muthhb = X_muthhb.cuda()
    #     X_ori = X_ori.cuda(); X_orihhb = X_orihhb.cuda()
    #     X_na = X_na.cuda()
    #     y = y.cuda()
    #     preds = model(X_ori, X_mut, X_orihhb, X_muthhb, X_na)
    #     loss = loss_fn(preds.ravel(), y)
    #     test_result.append(y.tolist())
    #     test_result.append(preds.squeeze(1).tolist())
    #     test_mse += loss.item()*len(y)
    # test_mse = test_mse/len(test_dataset)
    # test_rmse = math.sqrt(test_mse)
    # test_result.append(test_rmse)

    # file_name = 'result_rna/blstm_test_result.txt'
    # f = open(file_name,'w')
    # for i in test_result:
    #     f.write(str(i).strip('[').strip(']')+'\n')
    # f.close()
    # print(test_rmse)

if __name__ == "__main__":
    main()