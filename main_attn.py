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

nhid = 512 * 4
corpus = data.Corpus()
ntokens = len(data.corpus.dictionary_in)

batch_size = utils.args.batch_size
train_data_in = corpus.train_in.to(device)
print("train_data_in", train_data_in.shape)
train_data_out = corpus.train_out.to(device)
print("train_data_out", train_data_out.shape)
val_data_in = corpus.valid_in.to(device)
print("val_data_in", val_data_in.shape)
val_data_out = corpus.valid_out.to(device)
print("val_data_out", val_data_out.shape)

test_data_in = corpus.test_in.to(device)
print("test_data_in", test_data_in.shape)
test_data_out = corpus.test_out.to(device)
print("test_data_out", test_data_out.shape)

epochs = utils.args.epoch
lr = utils.args.lr
best_val_loss = None
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
    criterion = nn.NLLLoss()

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

hidden_size = 256
encoder = model_attn.EncoderRNN(data.corpus.dictionary_in.n_word, hidden_size).to(device)
attn_decoder = model_attn.AttnDecoderRNN(hidden_size, data.corpus.dictionary_out.n_word, dropout_p=0.1).to(device)

#trainIters(encoder, attn_decoder, print_every=50)


####################################################
encoder = model_attn.EncoderRNN(data.corpus.dictionary_in.n_word, hidden_size).to(device)
encoder.load_state_dict(torch.load(os.path.join(utils.saved_model, "encoder_weights.pt")))
encoder.eval()

attn_decoder = model_attn.AttnDecoderRNN(hidden_size, data.corpus.dictionary_out.n_word, dropout_p=0.1).to(device)
attn_decoder.load_state_dict(torch.load(os.path.join(utils.saved_model, "decoder_weights.pt")))
attn_decoder.eval()

def evaluate(encoder, decoder, sentence):
    with torch.no_grad():
        input_tensor = torch.tensor(sentence, dtype=torch.long, device=utils.device).view(-1, 1)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(utils.args.sent_maxlen, encoder.hidden_size, device=utils.device)
        
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[data.SOS_token]], device=utils.device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(utils.args.sent_maxlen, utils.args.sent_maxlen)

        for di in range(utils.args.sent_maxlen):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == data.EOS_token:
                decoded_words.append('EOS')
                break
            else:
                decoded_words.append(data.corpus.dictionary_out.idx2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(encoder, decoder, n=10):
    rand_nums = random.sample(range(test_data_in.size(0)), n)
    for i in range(n):
        sample = rand_nums.pop()
        input_data = test_data_in[sample].tolist()
        target_data = test_data_out[sample].tolist()
        output_words, attentions = evaluate(encoder, decoder, input_data)

        input_words = []
        target_words = []
        for idx in range(utils.args.sent_maxlen):
          input_words.append(data.corpus.dictionary_in.idx2word[input_data.pop(0)])
          target_words.append(data.corpus.dictionary_out.idx2word[target_data.pop(0)])
        input_sent = ' '.join(input_words)
        target_sent = ' '.join(target_words)
        print('>', input_sent)
        print('=', target_sent)
        
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

evaluateRandomly(encoder, attn_decoder)