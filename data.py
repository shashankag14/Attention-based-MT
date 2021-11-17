########################################################################
# Corpus data handling
########################################################################
import torch
import os
import numpy as np
from io import open
from sklearn.model_selection import train_test_split

path = 'data/cs-en.txt/'

## Check with the below code if the path is correct
# from os import walk
#
# filenames = next(walk(path), (None, None, []))[2]
# print(filenames)

########################################################################
# DICT
########################################################################
# def truncate_sentence(path, padding_token, max_words):
#     with open(path, 'r', encoding="utf8") as f:
#         for line in f:
#             words = line.split()
#             print(len(words))
#             if len(words) > max_words:
#                 print(words[:max_words])
#                 return words[:max_words]
#             words.extend([padding_token] * (max_words - len(words)))
#             print(words)
#             return words
# truncate_sentence(os.path.join(path, 'test_truncate.txt'), '<pad>', 4)

def truncate_sentence(words, max_words, padding_token):
    # print(len(words))
    if len(words) > max_words:
        # print(words[:max_words])
        words = words[:max_words]
    words.extend([padding_token] * (max_words - len(words)))
    # print(words)
    return words

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def add_all_words(self, path):
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split()
                # Make the size of all sentences same by truncating large sentences and adding <pad> to shorter ones
                truncated_words = truncate_sentence(words, 5, '<pad>')
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


data_cs = np.loadtxt(os.path.join(path, 'PHP.cs-en.cs'), dtype=str, delimiter='\n')
data_en = np.loadtxt(os.path.join(path, 'PHP.cs-en.en'), dtype=str, delimiter='\n')

train_cs, valid_cs = train_test_split(data_cs, test_size=0.3,
                                      random_state=42)  # train : 26384, valid : 6597 (Ratio - 0.8:0.2)
train_en, valid_en = train_test_split(data_en, test_size=0.3,
                                      random_state=42)  # train : 26384, valid : 6597 (Ratio - 0.8:0.2)

np.savetxt(os.path.join(path, 'train_in.txt'), train_cs, fmt='%s')  # cs
np.savetxt(os.path.join(path, 'valid_in.txt'), valid_cs, fmt='%s')  # cs
np.savetxt(os.path.join(path, 'train_out.txt'), train_en, fmt='%s')  # en
np.savetxt(os.path.join(path, 'valid_out.txt'), valid_en, fmt='%s')  # en
print("Train and validation data created and saved in ", path)

print("Length of Train and validation set in source language:", len(train_cs), len(valid_cs))
print("Length of Train and validation set in target language:", len(train_en), len(valid_en))

class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.dictionary.add_word('<pad>')
        self.dictionary.add_all_words(os.path.join(path, 'train_in.txt'))
        self.dictionary.add_all_words(os.path.join(path, 'train_out.txt'))
        self.dictionary.add_all_words(os.path.join(path, 'valid_in.txt'))
        self.dictionary.add_all_words(os.path.join(path, 'valid_out.txt'))

        self.train_in = self.tokenize(os.path.join(path, 'train_in.txt'))
        self.train_out = self.tokenize(os.path.join(path, 'train_out.txt'))
        self.valid_in = self.tokenize(os.path.join(path, 'valid_in.txt'))
        self.valid_out = self.tokenize(os.path.join(path, 'valid_out.txt'))

    def tokenize(self, path):
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split()
                truncated_words = truncate_sentence(words, 5, '<pad>')
                ids = []
                for word in truncated_words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.unsqueeze(torch.tensor(ids).type(torch.int64), dim=0))
            ids = torch.cat(idss, dim=0)
        return ids
