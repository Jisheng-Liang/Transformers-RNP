import torch
from torch import nn
from torch.nn import functional as F
from emden import TransformerEncoder

class MLPRegressor(nn.Module):
    def __init__(self):
        super(MLPRegressor, self).__init__()
        self.drop_prob = 0.25

        self.hidden1 = nn.Linear(20,256); self.hidden2 = nn.Linear(20,256)
        self.hidden3 = nn.Linear(30,256); self.hidden4 = nn.Linear(30,256)
        self.hidden5 = nn.Linear(4,256)

        self.hidden11 = nn.Linear(256,16); self.hidden22 = nn.Linear(256,16)
        self.hidden33 = nn.Linear(256,16); self.hidden44 = nn.Linear(256,16)
        self.hidden55 = nn.Linear(256,16)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=self.drop_prob)

        self.linear1 = nn.Linear(65600, 4096)
        self.linear2 = nn.Linear(4096, 2048)
        self.linear3 = nn.Linear(2048, 1024)
        self.linear4 = nn.Linear(1024, 256)
        self.linear5 = nn.Linear(256, 1)


    def forward(self, X_ori, X_mut, X_orihhb, X_muthhb, X_na):

        X_ori = self.hidden1(X_ori); X_ori = self.act(X_ori); X_ori = self.dropout(X_ori)
        X_mut = self.hidden2(X_mut); X_mut = self.act(X_mut); X_mut = self.dropout(X_mut)
        X_orihhb = self.hidden3(X_orihhb); X_orihhb = self.act(X_orihhb); X_orihhb = self.dropout(X_orihhb)
        X_muthhb = self.hidden4(X_muthhb); X_muthhb = self.act(X_muthhb); X_muthhb = self.dropout(X_muthhb)
        X_na = self.hidden5(X_na); X_na = self.act(X_na); X_na = self.dropout(X_na)

        X_ori = self.hidden11(X_ori); X_ori = self.act(X_ori)
        X_mut = self.hidden22(X_mut); X_mut = self.act(X_mut)
        X_orihhb = self.hidden33(X_orihhb); X_orihhb = self.act(X_orihhb)
        X_muthhb = self.hidden44(X_muthhb); X_muthhb = self.act(X_muthhb)
        X_na = self.hidden55(X_na); X_na = self.act(X_na)

        output = torch.cat((X_ori, X_mut, X_orihhb, X_muthhb, X_na),1)
        output = torch.flatten(output,1,2)
        output = self.linear1(output); output = self.act(output); output = self.dropout(output)
        output = self.linear2(output); output = self.act(output)
        output = self.linear3(output); output = self.act(output)
        output = self.linear4(output); output = self.act(output)
        output = self.linear5(output)

        return output

class CNNRegressor(nn.Module):
    def __init__(self):
        super(CNNRegressor, self).__init__()
        self.drop_prob = 0.1

        self.hidden1 = nn.Conv2d(1,256,3); self.hidden2 = nn.Conv2d(1,256,3)
        self.hidden3 = nn.Conv2d(1,256,3); self.hidden4 = nn.Conv2d(1,256,3)
        self.hidden5 = nn.Conv2d(1,256,1)

        self.avgpool = nn.AvgPool2d(2,stride=2)

        self.bn1 = nn.BatchNorm2d(256); self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(256); self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(256)

        self.hidden11 = nn.Conv2d(256,16,3); self.hidden22 = nn.Conv2d(256,16,3)
        self.hidden33 = nn.Conv2d(256,16,3); self.hidden44 = nn.Conv2d(256,16,3)
        self.hidden55 = nn.Conv2d(256,16,1)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=self.drop_prob)

        self.linear1 = nn.Linear(71824,8192)
        self.linear2 = nn.Linear(8192,2048)
        self.linear3 = nn.Linear(2048,1024)
        self.linear4 = nn.Linear(1024,256)
        self.linear5 = nn.Linear(256,1)


    def forward(self, X_ori, X_mut, X_orihhb, X_muthhb, X_na):
        X_ori = X_ori.unsqueeze(1); X_mut = X_mut.unsqueeze(1)
        X_orihhb = X_orihhb.unsqueeze(1); X_muthhb = X_muthhb.unsqueeze(1)
        X_na = X_na.unsqueeze(1)
        X_ori = self.hidden1(X_ori); X_ori = self.act(X_ori); X_ori = self.avgpool(X_ori); X_ori = self.bn1(X_ori)
        X_mut = self.hidden2(X_mut); X_mut = self.act(X_mut); X_mut = self.avgpool(X_mut); X_mut = self.bn2(X_mut)
        X_orihhb = self.hidden3(X_orihhb); X_orihhb = self.act(X_orihhb); X_orihhb = self.avgpool(X_orihhb); X_orihhb = self.bn3(X_orihhb)
        X_muthhb = self.hidden4(X_muthhb); X_muthhb = self.act(X_muthhb); X_muthhb = self.avgpool(X_muthhb); X_muthhb = self.bn4(X_muthhb)
        X_na = self.hidden5(X_na); X_na = self.act(X_na); X_na = self.avgpool(X_na); X_na = self.bn5(X_na)

        X_ori = self.hidden11(X_ori); X_ori = self.act(X_ori); X_ori = self.avgpool(X_ori)
        X_mut = self.hidden22(X_mut); X_mut = self.act(X_mut); X_mut = self.avgpool(X_mut)
        X_orihhb = self.hidden33(X_orihhb); X_orihhb = self.act(X_orihhb); X_orihhb = self.avgpool(X_orihhb)
        X_muthhb = self.hidden44(X_muthhb); X_muthhb = self.act(X_muthhb); X_muthhb = self.avgpool(X_muthhb)
        X_na = self.hidden55(X_na); X_na = self.act(X_na); X_na = self.avgpool(X_na)

        X_ori = torch.flatten(X_ori,1,3); X_mut = torch.flatten(X_mut,1,3)
        X_orihhb = torch.flatten(X_orihhb,1,3); X_muthhb = torch.flatten(X_muthhb,1,3)
        X_na = torch.flatten(X_na,1,3)

        output = torch.cat((X_ori, X_mut, X_orihhb, X_muthhb, X_na),1)
        
        output = self.linear1(output); output = self.act(output); output = self.dropout(output)
        output = self.linear2(output); output = self.act(output)
        output = self.linear3(output); output = self.act(output)
        output = self.linear4(output); output = self.act(output)
        output = self.linear5(output)

        return output

class LSTMRegressor(nn.Module):
    def __init__(self):
        super(LSTMRegressor, self).__init__()
        self.input_size_pro = 20
        self.input_size_prohhb = 30
        self.input_size_na = 4
        self.hidden_size = 256
        self.num_layers = 2
        self.output_size = 16
        self.num_directions = 1
        self.drop_prob = 0.1
        self.lstm_ori = nn.LSTM(self.input_size_pro, self.hidden_size, self.num_layers, dropout=self.drop_prob)
        self.lstm_mut = nn.LSTM(self.input_size_pro, self.hidden_size, self.num_layers, dropout=self.drop_prob)
        self.lstm_orihhb = nn.LSTM(self.input_size_prohhb, self.hidden_size, self.num_layers, dropout=self.drop_prob)
        self.lstm_muthhb = nn.LSTM(self.input_size_prohhb, self.hidden_size, self.num_layers, dropout=self.drop_prob)
        self.lstm_na = nn.LSTM(self.input_size_na, self.hidden_size, self.num_layers, dropout=self.drop_prob)
        self.linear1 = nn.Linear(self.hidden_size, self.output_size)
        self.linear2 = nn.Linear(self.hidden_size, self.output_size)
        self.linear3 = nn.Linear(self.hidden_size, self.output_size)
        self.linear4 = nn.Linear(self.hidden_size, self.output_size)
        self.linear5 = nn.Linear(self.hidden_size, self.output_size)
        self.act = nn.ReLU()

        self.linear6 = nn.Linear(65600, 8192); self.dropout = nn.Dropout(p=self.drop_prob)
        self.linear7 = nn.Linear(8192, 2048)
        self.linear8 = nn.Linear(2048, 1024)
        self.linear9 = nn.Linear(1024, 256)
        self.linearout = nn.Linear(256, 1)

    def forward(self, X_ori, X_mut, X_orihhb, X_muthhb, X_na):
        batch_size = X_ori.shape[0]
        X_ori = X_ori.permute(1,0,2); X_mut = X_mut.permute(1,0,2)
        X_orihhb = X_orihhb.permute(1,0,2); X_muthhb = X_muthhb.permute(1,0,2)
        X_na = X_na.permute(1,0,2)

        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).cuda()
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).cuda()
        o_ori, _ = self.lstm_ori(X_ori, (h_0, c_0)); o_ori = self.linear1(o_ori); o_ori = self.act(o_ori)

        hh_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).cuda()
        cc_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).cuda()
        o_mut, _ = self.lstm_mut(X_mut, (hh_0, cc_0)); o_mut = self.linear2(o_mut); o_mut = self.act(o_mut)


        hhhb_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).cuda() 
        chhb_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).cuda()
        o_orihhb, _ = self.lstm_orihhb(X_orihhb, (hhhb_0, chhb_0)); o_orihhb = self.linear3(o_orihhb); o_orihhb = self.act(o_orihhb)

        hhhhb_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).cuda()
        cchhb_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).cuda()
        o_muthhb, _ = self.lstm_muthhb(X_muthhb, (hhhhb_0, cchhb_0)); o_muthhb = self.linear4(o_muthhb); o_muthhb = self.act(o_muthhb)

        H_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).cuda()
        C_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).cuda()
        o_na, _ = self.lstm_na(X_na, (H_0, C_0)); o_na = self.linear5(o_na); o_na = self.act(o_na)
        output = torch.cat((o_ori, o_mut, o_orihhb, o_muthhb, o_na),0)
        output = output.permute(1,0,2)
        # out = torch.squeeze(out, 1)
        preds = torch.flatten(output, 1, 2)
        # print(preds.shape)
        preds = self.linear6(preds); preds = self.act(preds); preds = self.dropout(preds)
        preds = self.linear7(preds); preds = self.act(preds)
        # preds = self.linear77(preds); preds = self.act(preds)
        preds = self.linear8(preds); preds = self.act(preds)
        preds = self.linear9(preds); preds = self.act(preds)
        preds = self.linearout(preds)
        # preds = torch.squeeze(preds,1)
        return preds

class BLSTMRegressor(nn.Module):
    def __init__(self):
        super(BLSTMRegressor, self).__init__()
        self.input_size_pro = 20
        self.input_size_prohhb = 30
        self.input_size_na = 4
        self.hidden_size = 256
        self.num_layers = 2
        self.output_size = 16
        self.num_directions = 2
        self.drop_prob = 0.25
        self.lstm_mut = nn.LSTM(self.input_size_pro, self.hidden_size, self.num_layers, dropout=self.drop_prob, bidirectional = True)
        self.lstm_ori = nn.LSTM(self.input_size_pro, self.hidden_size, self.num_layers, dropout=self.drop_prob, bidirectional = True)
        self.lstm_muthhb = nn.LSTM(self.input_size_prohhb, self.hidden_size, self.num_layers, dropout=self.drop_prob, bidirectional = True)
        self.lstm_orihhb = nn.LSTM(self.input_size_prohhb, self.hidden_size, self.num_layers, dropout=self.drop_prob, bidirectional = True)
        self.lstm_na = nn.LSTM(self.input_size_na, self.hidden_size, self.num_layers, dropout=self.drop_prob, bidirectional = True)
        self.linear1 = nn.Linear(self.hidden_size*self.num_directions, self.output_size)
        self.linear2 = nn.Linear(self.hidden_size*self.num_directions, self.output_size)
        self.linear3 = nn.Linear(self.hidden_size*self.num_directions, self.output_size)
        self.linear4 = nn.Linear(self.hidden_size*self.num_directions, self.output_size)
        self.linear5 = nn.Linear(self.hidden_size*self.num_directions, self.output_size)
        self.act = nn.ReLU()

        self.linear6 = nn.Linear(65600, 8192); self.dropout = nn.Dropout(p=self.drop_prob)
        self.linear7 = nn.Linear(8192, 2048)
        self.linear8 = nn.Linear(2048, 1024)
        self.linear9 = nn.Linear(1024, 256)
        self.linearout = nn.Linear(256, 1)

    def forward(self, X_ori, X_mut, X_orihhb, X_muthhb, X_na):
        batch_size = X_ori.shape[0]
        X_ori = X_ori.permute(1,0,2); X_mut = X_mut.permute(1,0,2)
        X_orihhb = X_orihhb.permute(1,0,2); X_muthhb = X_muthhb.permute(1,0,2)
        X_na = X_na.permute(1,0,2)

        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).cuda()
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).cuda()
        o_ori, _ = self.lstm_ori(X_ori, (h_0, c_0)); o_ori = self.linear1(o_ori); o_ori = self.act(o_ori)

        hh_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).cuda()
        cc_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).cuda()
        o_mut, _ = self.lstm_mut(X_mut, (hh_0, cc_0)); o_mut = self.linear2(o_mut); o_mut = self.act(o_mut)


        hhhb_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).cuda()
        chhb_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).cuda()
        o_orihhb, _ = self.lstm_orihhb(X_orihhb, (hhhb_0, chhb_0)); o_orihhb = self.linear3(o_orihhb); o_orihhb = self.act(o_orihhb)

        hhhhb_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).cuda()
        cchhb_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).cuda()
        o_muthhb, _ = self.lstm_muthhb(X_muthhb, (hhhhb_0, cchhb_0)); o_muthhb = self.linear4(o_muthhb); o_muthhb = self.act(o_muthhb)

        H_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).cuda()
        C_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).cuda()
        o_na, _ = self.lstm_na(X_na, (H_0, C_0)); o_na = self.linear5(o_na); o_na = self.act(o_na)
        output = torch.cat((o_ori, o_mut, o_orihhb, o_muthhb, o_na),0)
        output = output.permute(1,0,2)
        # out = torch.squeeze(out, 1)
        preds = torch.flatten(output, 1, 2)
        # print(preds.shape)
        preds = self.linear6(preds); preds = self.act(preds); preds = self.dropout(preds)
        preds = self.linear7(preds); preds = self.act(preds)

        preds = self.linear8(preds); preds = self.act(preds)
        preds = self.linear9(preds); preds = self.act(preds)
        preds = self.linearout(preds)
        # preds = torch.squeeze(preds,1)
        return preds

class Trans(nn.Module):
    def __init__(self, n_layers=2, nhead=8, pro_model=20, hhb_model=30, na_model=4, embed_dim=256, dropout=0.25):
        super(Trans, self).__init__()

        # TransformerEncoder
        self.transencoder1 = TransformerEncoder(n_layers, vocab_size=pro_model, embed_dim=embed_dim, num_heads=nhead)
        self.transencoder2 = TransformerEncoder(n_layers, vocab_size=pro_model, embed_dim=embed_dim, num_heads=nhead)
        self.transencoder3 = TransformerEncoder(n_layers, vocab_size=hhb_model, embed_dim=embed_dim, num_heads=nhead)
        self.transencoder4 = TransformerEncoder(n_layers, vocab_size=hhb_model, embed_dim=embed_dim, num_heads=nhead)
        self.transencoder5 = TransformerEncoder(n_layers, vocab_size=na_model, embed_dim=embed_dim, num_heads=nhead)
        
        # deco
        self.deco1 = nn.Linear(embed_dim,16)
        self.deco11 = nn.Linear(embed_dim,16)
        self.deco2 = nn.Linear(embed_dim,16)
        self.deco22 = nn.Linear(embed_dim,16)
        self.deco3 = nn.Linear(embed_dim,32)

        # act
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        
        # MLP
        self.linear1 = nn.Linear(67200, 4096)
        self.linear2 = nn.Linear(4096, 2048)
        self.linear3 = nn.Linear(2048, 1024)
        self.linear4 = nn.Linear(1024, 256)
        self.linear5 = nn.Linear(256, 1)
        
    
    def forward(self, X_ori, X_mut, X_orihhb, X_muthhb, X_na, mask=None):
        # X_ori, X_mut
        X_ori = self.transencoder1(X_ori, mask=mask)
        X_mut = self.transencoder2(X_mut, mask=mask)
        # X_orihhb, X_muthhb
        X_orihhb = self.transencoder3(X_orihhb, mask=mask)
        X_muthhb = self.transencoder4(X_muthhb, mask=mask)
        # X_na
        X_na = self.transencoder5(X_na, mask=mask)

        # decoder, act
        X_ori = self.deco1(X_ori); X_mut = self.deco11(X_mut)
        X_orihhb = self.deco2(X_orihhb); X_muthhb = self.deco22(X_muthhb)
        X_na = self.deco3(X_na)
        X_ori = self.act(X_ori); X_mut = self.act(X_mut)
        X_orihhb = self.act(X_orihhb); X_muthhb = self.act(X_muthhb)
        X_na = self.act(X_na)

        # flatten
        X_ori = torch.flatten(X_ori,1,2); X_mut = torch.flatten(X_mut,1,2)
        X_orihhb = torch.flatten(X_orihhb,1,2); X_muthhb = torch.flatten(X_muthhb,1,2)
        X_na = torch.flatten(X_na,1,2)
        output = torch.concat((X_ori, X_mut, X_orihhb, X_muthhb, X_na), 1)

        # MLP
        output = self.linear1(output); output = self.act(output); output = self.dropout(output)
        output = self.linear2(output); output = self.act(output)
        output = self.linear3(output); output = self.act(output); output = self.dropout(output)
        output = self.linear4(output); output = self.act(output)
        output = self.linear5(output)
        return output
    
class Trans_vec(nn.Module):
    def __init__(self, n_layers=2, nhead=8, pro_model=20, hhb_model=30, na_model=4, embed_dim=256, dropout=0.25):
        super(Trans_vec, self).__init__()

        # TransformerEncoder
        self.transencoder1 = TransformerEncoder(n_layers, vocab_size=pro_model, embed_dim=embed_dim, num_heads=nhead)
        self.transencoder2 = TransformerEncoder(n_layers, vocab_size=pro_model, embed_dim=embed_dim, num_heads=nhead)
        self.transencoder3 = TransformerEncoder(n_layers, vocab_size=hhb_model, embed_dim=embed_dim, num_heads=nhead)
        self.transencoder4 = TransformerEncoder(n_layers, vocab_size=hhb_model, embed_dim=embed_dim, num_heads=nhead)
        self.transencoder5 = TransformerEncoder(n_layers, vocab_size=na_model, embed_dim=embed_dim, num_heads=nhead)
        
        # deco
        self.deco1 = nn.Linear(embed_dim,16)
        self.deco11 = nn.Linear(embed_dim,16)
        self.deco2 = nn.Linear(embed_dim,16)
        self.deco22 = nn.Linear(embed_dim,16)
        self.deco3 = nn.Linear(embed_dim,32)

        # act
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        
        # MLP
        self.linear1 = nn.Linear(67200, 4096)
        self.linear2 = nn.Linear(4096, 2048)
        self.linear3 = nn.Linear(2048, 1024)
        self.linear4 = nn.Linear(1024, 256)
        self.linear5 = nn.Linear(256, 1)
        
    
    def forward(self, X_ori, X_mut, X_orihhb, X_muthhb, X_na, mask=None):
        # X_ori, X_mut
        X_ori = self.transencoder1(X_ori, mask=mask)
        X_mut = self.transencoder2(X_mut, mask=mask)
        # X_orihhb, X_muthhb
        X_orihhb = self.transencoder3(X_orihhb, mask=mask)
        X_muthhb = self.transencoder4(X_muthhb, mask=mask)
        # X_na
        X_na = self.transencoder5(X_na, mask=mask)

        # decoder, act
        X_ori = self.deco1(X_ori); X_mut = self.deco11(X_mut)
        X_orihhb = self.deco2(X_orihhb); X_muthhb = self.deco22(X_muthhb)
        X_na = self.deco3(X_na)
        X_ori = self.act(X_ori); X_mut = self.act(X_mut)
        X_orihhb = self.act(X_orihhb); X_muthhb = self.act(X_muthhb)
        X_na = self.act(X_na)

        # flatten
        X_ori = torch.flatten(X_ori,1,2); X_mut = torch.flatten(X_mut,1,2)
        X_orihhb = torch.flatten(X_orihhb,1,2); X_muthhb = torch.flatten(X_muthhb,1,2)
        X_na = torch.flatten(X_na,1,2)
        output = torch.concat((X_ori, X_mut, X_orihhb, X_muthhb, X_na), 1)

        # MLP
        output = self.linear1(output); output = self.act(output); output = self.dropout(output)
        output = self.linear2(output); output = self.act(output)
        output = self.linear3(output); output = self.act(output); output = self.dropout(output)
        output = self.linear4(output); output = self.act(output)
        # output = self.linear5(output)
        return output