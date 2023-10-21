import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from dataset import pro_encoder, hhblits_encoder, na_encoder, pdb_encoder
from model import Trans_pdb
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
    
    batch_size = 32
    root_dir = "data/"
    hhblits_dir = root_dir+"hhblits/"
    dataset = []
    for i in ['test']:
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

    test_dataset = dataset[0]
    print(len(test_dataset))
    test_sampler = DistributedSampler(test_dataset)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, sampler=test_sampler, pin_memory=True)
    print(len(test_loader))
    loss_fn = nn.MSELoss()
    model = Trans_pdb().to(device)
    model = DDP(model,device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    optimizer = Adam(model.parameters(), lr=1e-3)

    # test
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

    test_mse = 0.
    test_result = []
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
        test_mse += loss.item()*len(y)
    test_mse = test_mse/len(test_dataset)
    test_rmse = math.sqrt(test_mse)

    file_name = 'result/trans_test_pdb.txt'
    f = open(file_name,'w')
    for i in test_result:
        f.write(str(i)+'\n')
    f.close()
    print(test_rmse)

if __name__ == "__main__":
    main()
