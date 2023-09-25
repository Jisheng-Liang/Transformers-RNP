import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from dataset import pro_encoder, hhblits_encoder, na_encoder
from model import *
import pandas as pd
import math

import torch.distributed as dist
import argparse
import os

from torch.optim import Adam
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    args = parser.parse_args()

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device=torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')

    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")

    epochs = 500
    batch_size = 32
    root_dir = "./"
    hhblits_dir = root_dir+"new/"

    dataset = []
    for i in ['train','val','test']:
        filename = root_dir+i+'.csv'
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
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    test_sampler = DistributedSampler(test_dataset)
    train_loader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size, sampler=train_sampler, pin_memory=True)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, sampler=val_sampler, pin_memory=True)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, sampler=test_sampler, pin_memory=True)
    print(len(train_loader),len(val_loader), len(test_loader))
    loss_fn = nn.MSELoss()
    model = Trans().to(device)
    model = DDP(model,device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.1)

    # train
    best_val_loss = float("inf")
    log = []
    for epoch in range(1, epochs+1):
        # train
        total_loss = 0.
        model.train()
        train_sampler.set_epoch(epoch)
        for X_ori, X_mut, X_orihhb, X_muthhb, X_na, y in train_loader:
            optimizer.zero_grad()
            X_ori = X_ori.to(device)
            X_mut = X_mut.to(device)
            X_muthhb = X_muthhb.to(device)
            X_orihhb = X_orihhb.to(device)
            X_na = X_na.to(device)
            y_preds = model(X_ori, X_mut, X_orihhb, X_muthhb, X_na, device)
            y = y.to(device)

            loss = loss_fn(y_preds.ravel(), y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()*len(X_ori)
        total_loss = torch.tensor(total_loss).to(device)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        total_mse = total_loss/len(train_dataset)
        # validate
        model.eval()
        val_losses = 0.
        with torch.no_grad():
            for X_ori, X_mut, X_orihhb, X_muthhb, X_na, y in val_loader:
                X_ori = X_ori.to(device)
                X_mut = X_mut.to(device)
                X_muthhb = X_muthhb.to(device)
                X_orihhb = X_orihhb.to(device)
                X_na = X_na.to(device)
                y = y.to(device)
                preds = model(X_ori, X_mut, X_orihhb, X_muthhb, X_na, device)
                val_loss = loss_fn(preds.ravel(), y)
                val_losses += val_loss.item()*len(X_ori)
        val_losses = torch.tensor(val_losses).to(device)
        dist.all_reduce(val_losses, op=dist.ReduceOp.SUM)
        val_mse = val_losses/len(val_dataset)
        

        log.append([epoch,total_mse.item(),val_mse.item()])
        print('| epoch {:3d} | lr {:02.5f} '
                      'train_loss {:5.5f} | val_loss {:5.5f} | val_rmse {:5.5f}'.
                      format(epoch, scheduler.get_last_lr()[0],total_mse, val_mse, math.sqrt(val_mse)))
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_model = model
            if epoch > 50:
                torch.save(best_model.state_dict(), 'result/trans_best_all.pth')
        scheduler.step()
    file_name = 'result/translog_' + str(rank) + '.txt'   
    f = open(file_name,'w')
    for i in log:
        f.write(str(i)+'\n')
    f.close()

    # test
    dist.barrier()
    test_mse = 0.
    test_result = []
    model.eval()
    for X_ori, X_mut, X_orihhb, X_muthhb, X_na, y in test_loader:
        X_mut = X_mut.to(device); X_muthhb = X_muthhb.to(device)
        X_ori = X_ori.to(device); X_orihhb = X_orihhb.to(device)
        X_na = X_na.to(device)
        y = y.to(device)
        preds = model(X_ori, X_mut, X_orihhb, X_muthhb, X_na, device)
        loss = loss_fn(preds.ravel(), y)
        
        test_result.append(y.tolist())
        test_result.append(preds.tolist())
        test_result.append(loss.tolist())
        test_loss += loss.item()*len(X_ori)
    test_loss = torch.tensor(test_loss).to(device)
    dist.all_reduce(test_loss, op=dist.ReduceOp.SUM)
    test_mse = test_loss/len(test_dataset)
    
    test_rmse = math.sqrt(test_mse.item())
    
    file_name = 'result/test'+str(rank)+'result.txt'
    f = open(file_name,'w')
    for i in test_result:
        f.write(str(i)+'\n')
    f.close()
    print(test_rmse)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()