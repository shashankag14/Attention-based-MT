import argparse
import math
import time
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda
path = 'data/cs-en.txt/'

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--data', type=str, default=path,
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=512,
                    help='size of word embeddings')
parser.add_argument('--lr', type=float, default=0.00002,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--batch_size', type=int, default=50, metavar='N',
                    help='batch size')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--epoch', type=int, default=10,
                    help='number of epoch for training')
parser.add_argument('--sent_maxlen', type=int, default=10,
                    help='maximum words in a sentence (remaining sentences will be removed from data)')

args = parser.parse_args([])


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))




