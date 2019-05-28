import os
import torch

class Dictionary(object):
    def __init__(self, path=None):
        self.word2idx = {}
        self.idx2word = []
        if path:
            self.load(path)

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def load(self, path):
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    word = word.lower()
                    self.dictionary.add_word(word)
            self.dictionary.add_word('unk')



class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))


    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        tokens = 0
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    word = word.lower()
                    self.dictionary.add_word(word)
            self.dictionary.add_word('unk')
        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    word = word.lower()
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids
