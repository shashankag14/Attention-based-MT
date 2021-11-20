########################################################################
# Model
########################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda

# class Encoder(nn.Module):
#     def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
#         super(Encoder, self).__init__()
#         self.dropout = nn.Dropout(p)
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#
#         self.embedding = nn.Embedding(input_size, embedding_size)
#         self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
#
#     def forward(self, x):
#         # x shape: (seq_length, N) where N is batch size
#         print("X shape in encoder :", x.shape)
#         embedding = self.dropout(self.embedding(x))
#         # embedding shape: (seq_length, N, embedding_size)
#
#         outputs, (hidden, cell) = self.rnn(embedding)
#         # outputs shape: (seq_length, N, hidden_size)
#         print("hidden and cell shape:", hidden.shape, cell.shape)
#         return hidden, cell
#
#
# class Decoder(nn.Module):
#     def __init__(
#         self, input_size, embedding_size, hidden_size, output_size, num_layers, p
#     ):
#         super(Decoder, self).__init__()
#         self.dropout = nn.Dropout(p)
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#
#         self.embedding = nn.Embedding(input_size, embedding_size)
#         self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x, hidden, cell):
#         # x shape: (N) where N is for batch size, we want it to be (1, N), seq_length
#         # is 1 here because we are sending in a single word and not a sentence
#         x = x.unsqueeze(0)
#         print("X shape :", x.shape)
#
#         embedding = self.dropout(self.embedding(x))
#         # embedding shape: (1, N, embedding_size)
#
#         outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
#         # outputs shape: (1, N, hidden_size)
#
#         predictions = self.fc(outputs)
#         print("predictions shape :", predictions.shape)
#         # predictions shape: (1, N, length_target_vocabulary) to send it to
#         # loss function we want it to be (N, length_target_vocabulary) so we're
#         # just gonna remove the first dim
#         predictions = predictions.squeeze(0)
#
#         return predictions, hidden, cell
#
#
# class Seq2Seq(nn.Module):
#     def __init__(self, encoder, decoder):
#         super(Seq2Seq, self).__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#
#     def forward(self, source, target, teacher_force_ratio=0.5):
#         print("source and target shape in seq2seq:", source.shape, target.shape)
#         batch_size = source.shape[1]
#         target_len = target.shape[0]
#         target_vocab_size = len(data.corpus.dictionary_out)
#
#         outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
#         print("outputs shape in eq2seq zeros :", outputs.shape)
#         hidden, cell = self.encoder(source)
#
#         # Grab the first input to the Decoder which will be <SOS> token
#         x = target[0]
#
#         for t in range(1, target_len):
#             # Use previous hidden, cell as context from encoder at start
#             output, hidden, cell = self.decoder(x, hidden, cell)
#
#             # Store next output prediction
#             outputs[t] = output
#
#             # Get the best word the Decoder predicted (index in the vocabulary)
#             best_guess = output.argmax(1)
#
#             # With probability of teacher_force_ratio we take the actual next word
#             # otherwise we take the word that the Decoder predicted it to be.
#             # Teacher Forcing is used so that the model gets used to seeing
#             # similar inputs at training and testing time, if teacher forcing is 1
#             # then inputs at test time might be completely different than what the
#             # network is used to. This was a long comment.
#             x = target[t] if random.random() < teacher_force_ratio else best_guess
#
#         return outputs

#
####################################################################
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
        # decoded = self.decoder(hidden)
        decoded = self.decoder(output)
        decoded_v = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded_v, dim=1)
# ####################################################################
# class RNNModel(nn.Module):
#     def __init__(self, ntoken, ninp, nhid):
#         super(RNNModel, self).__init__()
#         self.ntoken = ntoken
#         self.encoder = nn.Embedding(ntoken, ninp)
#         self.att_w_s = nn.Linear(512*4,1)
#         self.att_w_h = nn.Linear(512*4*2,1)
#         self.nhid=nhid
#         self.rnn_bi = nn.RNN(ninp, nhid,batch_first=True,bidirectional=True)
#         self.rnn_w = nn.Linear(nhid*3+ninp,nhid)
#         self.decoder = nn.Linear(nhid, ntoken)
#         self.init_weights()
#     def init_weights(self):
#         initrange = 0.1
#         nn.init.uniform_(self.encoder.weight, -initrange, initrange)
#         nn.init.zeros_(self.decoder.weight)
#         nn.init.uniform_(self.decoder.weight, -initrange, initrange)
#         nn.init.uniform_(self.att_w_s.weight, -initrange,initrange)
#         nn.init.uniform_(self.att_w_h.weight, -initrange,initrange)
#         nn.init.uniform_(self.rnn_w.weight, -initrange,initrange)
#
#     def forward(self, src, tgt):
#     emb_src = self.encoder(src)
#
#     emb_tgt = self.encoder(tgt)
#     bi_src, hiddenbi = self.rnn_bi(emb_src)
#     all_h = []
#     embs = torch.split(emb_tgt, 1, dim=1)
#     current_h = torch.zeros(src.shape[0], 1, self.nhid)
#     for i in range(len(embs)):
#         att = torch.sigmoid(torch.sum(self.att_w_s(torch.unsqueeze(current_h, d
#         im = 2))+self.att_w_h(torch.unsqueeze(bi_src, dim=1)), dim = 3, keepdim = False))
#         mask = -999999999 * F.relu(1 - src)
#         att = att + torch.unsqueeze(mask, dim=1)
#         att = F.softmax(att, dim=2)
#         current_c = torch.sum(torch.unsqueeze(att, dim=3) * torch.unsqueeze(bi_src, dim = 1), dim = 2)
#         current_h = torch.sigmoid(self.rnn_w(torch.cat([embs[i], current_h, current_c], dim=2)))
#         all_h.append(current_h)
#     output = torch.cat(all_h, dim=1)
#     decoded = self.decoder(output)
#     decoded = decoded.view(-1, self.ntoken)
#     return F.log_softmax(decoded, dim=1)