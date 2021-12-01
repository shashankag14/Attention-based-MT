########################################################################
# Corpus data handling
########################################################################
import torch
import os
import numpy as np
from io import open
from sklearn.model_selection import train_test_split
import unicodedata
import re

# local files in project
import utils

train_flag = 0
valid_flag = 1
test_flag = 2

train_data_path = os.path.join(utils.path, 'train_data')
valid_data_path = os.path.join(utils.path, 'valid_data')
test_data_path = os.path.join(utils.path, 'test_data')
if not os.path.isdir(train_data_path):
    os.mkdir(train_data_path)
if not os.path.isdir(valid_data_path):
    os.mkdir(valid_data_path)
if not os.path.isdir(test_data_path):
    os.mkdir(test_data_path)

########################################################################
# Splitting data into train/valid/test
########################################################################
data_cs = np.loadtxt(os.path.join(utils.path, 'PHP.cs-en.cs'), dtype=str, delimiter='\n')
data_en = np.loadtxt(os.path.join(utils.path, 'PHP.cs-en.en'), dtype=str, delimiter='\n')

train_cs, rem_cs = train_test_split(data_cs, train_size=0.9,
                                      random_state=27)  # train : 26384, remaining : 6597 (Ratio - 0.6:0.4)
train_en, rem_en = train_test_split(data_en, train_size=0.9,
                                      random_state=27)  # train : 26384, remaining : 6597 (Ratio - 0.6:0.4)

# Now since we want the valid and test size to be equal (10% each of overall data).
# we have to define valid_size=0.5 (that is 50% of remaining data)
valid_cs, test_cs = train_test_split(rem_cs, test_size=0.5,
                                      random_state=27)
valid_en, test_en = train_test_split(rem_en, test_size=0.5,
                                      random_state=27)

np.savetxt(os.path.join(train_data_path, 'train_in.txt'), train_cs, fmt='%s')  # cs
np.savetxt(os.path.join(valid_data_path, 'valid_in.txt'), valid_cs, fmt='%s')  # cs
np.savetxt(os.path.join(test_data_path, 'test_in.txt'), rem_cs, fmt='%s')  # cs

np.savetxt(os.path.join(train_data_path, 'train_out.txt'), train_en, fmt='%s')  # en
np.savetxt(os.path.join(valid_data_path, 'valid_out.txt'), valid_en, fmt='%s')  # en
np.savetxt(os.path.join(test_data_path, 'test_out.txt'), rem_en, fmt='%s')  # en

# print("Number of sentences of Train, validation and test set in source language:", len(train_cs), len(valid_cs), len(rem_cs))
# print("Number of sentences of Train, validation and test set in target language:", len(train_en), len(valid_en), len(rem_en))

########################################################################
# DATA PRE-PROC
########################################################################
def truncate_sentence(words, max_words, padding_token):
    if len(words) > max_words:
        words = words[:max_words]
    words.extend([padding_token] * (max_words - len(words)))
    # eos_str = 'EOS'
    # words.extend([eos_str])
    return words

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

########################################################################
# DICTIONARY
########################################################################
# pad_token  = 0
SOS_token = 0
EOS_token = 1

class Dictionary(object):
    def __init__(self):
        self.word2idx = {} # mapping word to its idx
        self.word2count = {} # frequency of each word
        self.idx2word = {SOS_token : 'SOS', EOS_token : 'EOS'} # mapping idx to its word
        self.n_word = 2 # pad, SOS and EOS # total number of unique words in dictionary

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.n_word
            self.word2count[word] = 1
            self.idx2word[self.n_word] = word
            self.n_word += 1
        else :
            self.word2count[word] += 1
        return self.word2idx[word]

    def add_all_words(self, path):
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                normalised_line = normalizeString(line)
                words = normalised_line.split()
                # Make the size of all sentences same by truncating large sentences and adding <pad> to shorter ones
                truncated_words = truncate_sentence(words, (utils.args.sent_maxlen-1), "<pad>")
                for word in truncated_words:
                    self.add_word(word)

    def getsentence(self, sent):
        ids = []
        words = sent.split()
        for word in words:
            ids.append(self.word2idx[word])
        return ids

    def __len__(self):
        return len(self.idx2word)

########################################################################
# DATA TOKENIZATION
########################################################################
class Corpus(object):
    def __init__(self):
        self.dictionary_in = Dictionary()
        self.dictionary_out = Dictionary()
        self.dictionary_in.add_all_words(os.path.join(train_data_path, 'train_in.txt'))
        #self.dictionary_in.add_all_words(os.path.join(valid_data_path, 'valid_in.txt'))
        self.dictionary_out.add_all_words(os.path.join(train_data_path, 'train_out.txt'))
        #self.dictionary_out.add_all_words(os.path.join(valid_data_path, 'valid_out.txt'))
  
        self.dictionary_in.add_all_words(os.path.join(test_data_path, 'test_in.txt'))
        self.dictionary_out.add_all_words(os.path.join(test_data_path, 'test_out.txt'))


        self.train_in = self.tokenize(os.path.join(train_data_path, 'train_in.txt'), self.dictionary_in)
        self.train_out = self.tokenize(os.path.join(train_data_path, 'train_out.txt'), self.dictionary_out)
        #self.valid_in = self.tokenize(os.path.join(valid_data_path, 'valid_in.txt'), self.dictionary_in)
        #self.valid_out = self.tokenize(os.path.join(valid_data_path, 'valid_out.txt'), self.dictionary_out)
        
        self.test_in = self.tokenize(os.path.join(test_data_path, 'test_in.txt'), self.dictionary_in)
        self.test_out = self.tokenize(os.path.join(test_data_path, 'test_out.txt'), self.dictionary_out)

    def tokenize(self, path, dictionary):
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                normalised_line = normalizeString(line)
                words = normalised_line.split()
                truncated_words = truncate_sentence(words, (utils.args.sent_maxlen-1), '<pad>')
                ids = []
                for word in truncated_words:
                    ids.append(dictionary.word2idx[word])
                ids.append(EOS_token)
                idss.append(torch.unsqueeze(torch.tensor(ids).type(torch.int64), dim=0))
            ids = torch.cat(idss, dim=0)
        return ids


corpus = Corpus()
