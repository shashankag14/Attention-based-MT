import torch.nn as nn
from io import open
import torch
import torch.optim
# import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import math
import random
import pandas as pd # for saving metrices in csv
#### For Pytorch guide implementation
# import matplotlib.pyplot as plt
# plt.switch_backend('agg')
# import matplotlib.ticker as ticker
import numpy as np
import os


# local files in project
import data
import model_attn
import utils

torch.manual_seed(utils.args.seed) # for reproducibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda

print("Number of sentences of Train, validation and test set in source language:", len(data.train_cs), len(data.valid_cs), len(data.rem_cs))
print("Number of sentences of Train, validation and test set in target language:", len(data.train_en), len(data.valid_en), len(data.rem_en))

print("\nLength of source dictionary : ", corpus.dictionary_in.n_word)
print("Length of target dictionary : ", corpus.dictionary_out.n_word)

batch_size = utils.args.batch_size
train_data_in = data.corpus.train_in.to(device)
print("train_data_in", train_data_in.shape)
train_data_out = data.corpus.train_out.to(device)
print("train_data_out", train_data_out.shape)

history = {'train_loss': [], 'val_loss': [], 'epoch': []}  # Collects per-epoch loss

teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=utils.args.sent_maxlen):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[data.SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == data.EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(encoder, decoder, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss(ignore_index = data.corpus.dictionary_out.word2idx['<pad>'])

    for iter in range(1, train_data_in.size(0)+1):
        # randomly sampling input and target sentences for training
        input_tensor = torch.tensor(train_data_in[iter-1].tolist(), dtype=torch.long, device=device).view(-1, 1)
        target_tensor = torch.tensor(train_data_out[iter-1].tolist(), dtype=torch.long, device=device).view(-1, 1)

        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d/%d  %d%%) %.4f' % (utils.timeSince(start, iter / train_data_in.size(0)), iter, train_data_in.size(0), iter / train_data_in.size(0) * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    pd.DataFrame.from_dict(data=plot_losses, orient='columns').to_csv(os.path.join(os.getcwd(),'train_loss.csv'))
    # showPlot(plot_losses)
    torch.save(encoder.state_dict(), os.path.join(utils.saved_model, "encoder_weights.pt"))
    torch.save(decoder.state_dict(), os.path.join(utils.saved_model, "decoder_weights.pt"))

encoder = model_attn.EncoderRNN(data.corpus.dictionary_in.n_word, utils.args.hsize).to(device)
attn_decoder = model_attn.AttnDecoderRNN(utils.args.hsize, data.corpus.dictionary_out.n_word, dropout_p=0.1).to(device)

trainIters(encoder, attn_decoder, print_every=50)
