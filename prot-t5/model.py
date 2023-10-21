import torch
from torch import nn
from torch.nn import functional as F
from emden import TransformerEncoder

class prott5(nn.Module):
    def __init__(self, n_layers=2, nhead=8, pro_model=1024, na_model=640, embed_dim=16, dropout=0.1):
        super(prott5, self).__init__()

        # TransformerEncoder
        self.protencoder = TransformerEncoder(n_layers, vocab_size=1000, embed_dim=embed_dim, num_heads=nhead, middle_dim=2048)
        self.rnaencoder = TransformerEncoder(n_layers, vocab_size=1000, embed_dim=embed_dim, num_heads=nhead, middle_dim=2048)

        # average pooling
        self.pooling = nn.AvgPool2d((3,3), stride=(2,2))

        # decoder
        self.decoder = nn.Linear(3843,1)
        
    def forward(self, pro, rna):

        # protein sequence
        pro_emb = self.protencoder(pro)
        pro_emb = self.pooling(pro_emb)

        # RNA sequence
        rna_emb = self.rnaencoder(rna)
        rna_emb = self.pooling(rna_emb)

        # FC layer
        pro_emb = torch.flatten(pro_emb,1,2); rna_emb = torch.flatten(rna_emb,1,2)

        output = torch.concat((pro_emb, rna_emb), 1)
        # decoder
        output = self.decoder(output)

        return output
    
class prot_only(nn.Module):
    def __init__(self, n_layers=2, nhead=8, pro_model=1024, na_model=640, embed_dim=256, mid_dim=1024, dropout=0.1):
        super(prot_only, self).__init__()

        # TransformerEncoder
        self.protencoder = TransformerEncoder(n_layers, vocab_size=pro_model,
         embed_dim=embed_dim, num_heads=nhead, middle_dim=mid_dim, drop_prob=dropout)
        self.oriencoder = TransformerEncoder(n_layers, vocab_size=pro_model,
         embed_dim=embed_dim, num_heads=nhead, middle_dim=mid_dim, drop_prob=dropout)
        self.rnaencoder = TransformerEncoder(n_layers, vocab_size=na_model,
         embed_dim=embed_dim, num_heads=nhead, middle_dim=mid_dim, drop_prob=dropout)
        # average pooling
        self.pooling = nn.AvgPool2d((5,5), stride=(5,5))

        # decoder
        self.pro_decoder = nn.Linear(256, 32)
        self.ori_decoder = nn.Linear(256, 32)
        self.rna_decoder = nn.Linear(256, 32)

        self.decoder1 = nn.Linear(67264,4096)
        self.act = nn.ReLU()
        self.decoder2 = nn.Linear(4096,512)
        self.decoder3 = nn.Linear(512,1)
        
    def forward(self, pro, ori, rna):

        # protein sequence
        pro_emb = self.protencoder(pro)
        ori_emb = self.protencoder(ori)
        rna_emb = self.rnaencoder(rna)
        
        # Linear
        pro_emb = self.pro_decoder(pro_emb); pro_emb = self.act(pro_emb)
        ori_emb = self.ori_decoder(ori_emb); ori_emb = self.act(ori_emb)
        rna_emb = self.rna_decoder(rna_emb); rna_emb = self.act(rna_emb)
        # # pooling
        # pro_emb = self.pooling(pro_emb)
        # ori_emb = self.pooling(ori_emb)
        # rna_emb = self.pooling(rna_emb)
        # FC layer
        pro_emb = torch.flatten(pro_emb,1,2); ori_emb = torch.flatten(ori_emb,1,2); rna_emb = torch.flatten(rna_emb,1,2)

        output = torch.concat((pro_emb, ori_emb, rna_emb), 1)
        # decoder
        output = self.decoder1(output); output = self.act(output)
        output = self.decoder2(output); output = self.act(output)
        output = self.decoder3(output)

        return output