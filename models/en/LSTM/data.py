import os
import torch
from .tokenizer import tokenize

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
    def __init__(self, path, language):
        self.dictionary = Dictionary()
        self.train = create_tokenized_tensor(tokenize(os.path.join(path, 'train.txt'), language, self.dictionary), self.dictionary)
        self.valid = create_tokenized_tensor(tokenize(os.path.join(path, 'valid.txt'), language, self.dictionary), self.dictionary)
        self.test = create_tokenized_tensor(tokenize(os.path.join(path, 'test.txt'), language, self.dictionary), self.dictionary)


def create_tokenized_tensor(iterator, dictionary):
    """Create tensor of embeddings from word iterator."""
    tensor = torch.LongTensor(len(iterator))
    token = 0
    for item in iterator:
        tensor[token] = dictionary.word2idx[item] if item in dictionary.word2idx else dictionary.word2idx['<unk>']
        token += 1
    return tensor
