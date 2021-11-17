import torch.nn as nn
from io import open
import argparse
import data
import time
import model
import torch
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='train')
parser.add_argument('--data', type=str, default=data.path,
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=1024,
                    help='size of word embeddings')
parser.add_argument('--lr', type=float, default=0.00002,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--batch_size', type=int, default=2, metavar='N',
                    help='batch size')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
# parser.add_argument('--max_words', type=int, default=4,
#                     help='maximum words in a sentence (remaining sentences will be removed from data)')


# parser.parse_args()

########################################################################
# TRAIN/EVAL
########################################################################
def train(train_data_in, train_data_out, batch_size):
    model.train()
    i = 0
    running_loss = 0
    while i + batch_size <= train_data_in.size(0):
        data = train_data_in[i:i + batch_size, :]
        targets = train_data_out[i:i + batch_size, :].view(-1)
        model.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        if (i + batch_size) % 100 == 0:
            print("Itr {}/{} Train Loss : {:.3f}".format(i + batch_size, train_data_in.size(0), loss.item()))
        running_loss += loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-lr)
        i += batch_size
    train_loss = running_loss / train_data_in.size(0)
    return train_loss


def evaluate(data_source_in, data_source_out, batch_size):
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        i = 0
        while i + batch_size <= data_source_in.size(0):
            data = data_source_in[i:i + batch_size, :]
            targets = data_source_out[i:i + batch_size, :].view(-1)
            output = model(data)
            total_loss += criterion(output, targets).item()
            i += batch_size
    return total_loss/data_source_in.size(0)

########################################################################
# train/eval loop
########################################################################
args = parser.parse_args([])
torch.manual_seed(args.seed)
device = torch.device('cpu')  # cuda
nhid = 512 * 4
corpus = data.Corpus(data.path)
ntokens = len(corpus.dictionary)
model = model.RNNModel(ntokens, args.emsize, nhid).to(device)
criterion = nn.NLLLoss(ignore_index=0)
batch_size = 5
train_data_in = corpus.train_in.to(device)
train_data_out = corpus.train_out.to(device)
val_data_in = corpus.valid_in.to(device)
val_data_out = corpus.valid_out.to(device)
epochs = 10
lr = args.lr
best_val_loss = None

history = {'train_loss': [], 'val_loss': [], 'epoch': []}  # Collects per-epoch loss

for epoch in range(1, epochs + 1):
    print("-"*30)
    print("Epoch {}/{}:".format(epoch, epochs))
    epoch_start_time = time.time()

    # Training for one epoch
    train_loss = train(train_data_in, train_data_out, batch_size)

    # Validation for one epoch
    val_loss = evaluate(val_data_in, val_data_out, batch_size)
    print("Epoch Validation loss :", val_loss)

    epoch_total_time = time.time() - epoch_start_time
    print("Epoch time : {} seconds".format(epoch_total_time))

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['epoch'].append(epoch)

    if not best_val_loss or val_loss < best_val_loss:
        with open(args.save, 'wb') as f:
            torch.save(model, f)
        best_val_loss = val_loss
        print("Best loss : ", best_val_loss)

    # Plot loss metrics
    train_loss_hist = dict['train_loss']
    val_loss_hist = dict['val_loss']
    plt.figure()
    plt.plot(epochs, train_loss_hist, 'b', label='Training loss')
    plt.plot(epochs, val_loss_hist, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

with open(args.save, 'rb') as f:
    model = torch.load(f).to(device)

model.eval()
corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)
input = corpus.dictionary.getsentence('a b c <eos>')
input_tensor = torch.unsqueeze(torch.tensor(input).type(torch.int64), dim=0).to(device)
outf = 'test.out'
with open(outf, 'w') as outf:
    with torch.no_grad():
        word = 'none'
        while word != '<eos>':
            output = model(input_tensor)
            word_idx = torch.argmax(output[-1, :]).cpu()
            input.append(word_idx)
            input_tensor = torch.unsqueeze(torch.tensor(input).type(torch.int64), dim=0).to(device)
            word = corpus.dictionary.idx2word[word_idx]
            outf.write(word)
