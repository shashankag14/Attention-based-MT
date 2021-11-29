########################################################################
# Model
########################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda

####################################################################
class RNNModel(nn.Module):
    def __init__(self, ntoken, ninp, nhid):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.encoder = nn.Embedding(ntoken, ninp)
        self.att_w_s = nn.Linear(512*4,1)
        self.att_w_h = nn.Linear(512*4*2,1)
        self.nhid=nhid
        self.rnn_bi = nn.RNN(ninp, nhid,batch_first=True,bidirectional=True)
        self.rnn_w = nn.Linear(nhid*3+ninp,nhid)
        self.decoder = nn.Linear(nhid, ntoken)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)
        nn.init.uniform_(self.att_w_s.weight, -initrange,initrange)
        nn.init.uniform_(self.att_w_h.weight, -initrange,initrange)
        nn.init.uniform_(self.rnn_w.weight, -initrange,initrange)

    def forward(self, src, tgt):
        emb_src = self.encoder(src)
        emb_tgt = self.encoder(tgt)
        bi_src, hiddenbi = self.rnn_bi(emb_src)
        all_h = []
        embs = torch.split(emb_tgt, 1, dim=1)
        current_h = torch.zeros(src.shape[0], 1, self.nhid)
        for i in range(len(embs)):
            att = torch.sigmoid(torch.sum(self.att_w_s(torch.unsqueeze(current_h, dim = 2))+self.att_w_h(torch.unsqueeze(bi_src, dim=1)), dim = 3, keepdim = False))
            mask = -999999999 * F.relu(1 - src)
            att = att + torch.unsqueeze(mask, dim=1)
            att = F.softmax(att, dim=2)
            current_c = torch.sum(torch.unsqueeze(att, dim=3) * torch.unsqueeze(bi_src, dim = 1), dim = 2)
            input_cat = torch.cat([embs[i], current_h, current_c], dim=2)
            current_h = torch.sigmoid(self.rnn_w(input_cat))
            all_h.append(current_h)
        output = torch.cat(all_h, dim=1)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1)