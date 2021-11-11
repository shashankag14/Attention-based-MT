########################################################################
# Model
########################################################################

import torch.nn as nn
import torch.nn.functional as F


class RNNModel(nn.Module):
    def __init__(self, ntoken, ninp, nhid):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.encoder = nn.Embedding(ntoken, 512)
        self.rnn = nn.RNN(512, 512 * 4, batch_first=True)
        self.decoder = nn.Linear(512 * 4, ntoken)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src):
        emb = self.encoder(src)
        output, hidden = self.rnn(emb)
        decoded = self.decoder(output)
        decoded_v = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded_v, dim=1)
