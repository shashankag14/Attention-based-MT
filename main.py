import torch.nn as nn
from io import open
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import math

# local files in project
import data
import model
import utils

########################################################################
# train/eval loop
########################################################################

torch.manual_seed(utils.args.seed) # for reproducibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda
nhid = 512 * 4
corpus = data.Corpus(data.train_flag)
ntokens = len(data.corpus.dictionary_in)
model = model.RNNModel(ntokens, utils.args.emsize, nhid).to(device)
criterion = nn.NLLLoss(ignore_index=data.pad_token)
batch_size = utils.args.batch_size
train_data_in = corpus.train_in.to(device)
print("train_data_in", train_data_in.shape)
train_data_out = corpus.train_out.to(device)
print("train_data_out", train_data_out.shape)
val_data_in = corpus.valid_in.to(device)
print("val_data_in", val_data_in.shape)
val_data_out = corpus.valid_out.to(device)
print("val_data_out", val_data_out.shape)
epochs = 10
lr = utils.args.lr
best_val_loss = None

history = {'train_loss': [], 'val_loss': [], 'epoch': []}  # Collects per-epoch loss
########################################################################
# TRAIN/EVAL
########################################################################
def train(train_data_in, train_data_out, batch_size):
    model.train()
    i = 0
    itr = 0
    total_train_loss = 0
    with tqdm(total = math.ceil(train_data_in.size(0)/batch_size)) as pbar:
        while i + batch_size <= train_data_in.size(0):
            itr += 1
            data = train_data_in[i:i + batch_size, :]
            targets = train_data_out[i:i + batch_size, :]#.view(-1) # changed this and moved to criterion

            model.zero_grad()
            output = model(data, targets)
            loss = criterion(output, targets.view(-1))

            if itr % 100 == 0:
                print("Itr {}/{} Train Loss : {:.3f}".format(itr, train_data_in.size(0), loss.item()))

            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), utils.args.clip)

            for p in model.parameters():
                p.data.add_(p.grad, alpha=-lr)
            i += batch_size
            pbar.update(1)
        pbar.close()

    train_loss = total_train_loss / train_data_in.size(0)
    return train_loss


def evaluate(data_source_in, data_source_out, batch_size):
    model.eval()
    total_val_loss = 0.
    with torch.no_grad():
        i = 0
        while i + batch_size <= data_source_in.size(0):
            data = data_source_in[i:i + batch_size, :]
            targets = data_source_out[i:i + batch_size, :]#.view(-1)

            output = model(data, targets)
            total_val_loss += criterion(output, targets.view(-1)).item()
            i += batch_size
    return total_val_loss/data_source_in.size(0)

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
        with open(utils.args.save, 'wb') as f:
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

with open(utils.args.save, 'rb') as f:
    model = torch.load(f).to(device)

model.eval()
corpus = data.Corpus(data.valid_flag)
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

