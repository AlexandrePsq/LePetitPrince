import os
import torch
from .tokenizer import tokenize
from collections import defaultdict
import logging

class Dictionary(object):
    def __init__(self, path, language):
        self.word2idx = {}
        self.idx2word = []
        self.language = language
        self.word2freq = defaultdict(int)

        vocab_path = os.path.join(path, 'vocab.txt')
        try:
            vocab = open(vocab_path, encoding="utf8").read()
            self.word2idx = {w: i for i, w in enumerate(vocab.split())}
            self.idx2word = [w for w in vocab.split()]
            self.vocab_file_exists = True
        except FileNotFoundError:
            logging.info("Vocab file not found, creating new vocab file.")
            self.create_vocab(os.path.join(path, 'train.txt'))
            open(vocab_path,"w").write("\n".join([w for w in self.idx2word]))


    def add_word(self, word):
        self.word2freq[word] += 1
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1


    def __len__(self):
        return len(self.idx2word)


    def create_vocab(self, path):
        iterator = tokenize(path, self.language, train=True)
        for item in iterator:
            self.add_word(item)
        self.add_word('<unk>')



class Corpus(object):
    def __init__(self, path, language):
        self.dictionary = Dictionary(path, language)
        train_path = os.path.join(path, 'train.txt')
        valid_path = os.path.join(path, 'valid.txt')
        test_path = os.path.join(path, 'test.txt')
        train_tensor = os.path.join(path, 'train.pkl')
        valid_tensor = os.path.join(path, 'valid.pkl')
        test_tensor = os.path.join(path, 'test.pkl')
        try:
            with open(train_tensor, 'rb') as f:
                self.train = torch.load(f)
            with open(valid_tensor, 'rb') as f:
                self.valid = torch.load(f)
            with open(test_tensor, 'rb') as f:
                self.test = torch.load(f)

        except FileNotFoundError:
            logging.info("Tensor files not found, creating new tensor files.")
            self.train = create_tokenized_tensor(tokenize(train_path, language, self.dictionary, train=True), self.dictionary)
            self.valid = create_tokenized_tensor(tokenize(valid_path, language, self.dictionary, train=True), self.dictionary)
            self.test = create_tokenized_tensor(tokenize(test_path, language, self.dictionary, train=True), self.dictionary)
            
            with open(train_tensor, 'wb') as f:
                torch.save(self.train, f)
            with open(valid_tensor, 'wb') as f:
                torch.save(self.valid, f)
            with open(test_tensor, 'wb') as f:
                torch.save(self.test, f)
        



def create_tokenized_tensor(iterator, dictionary):
    """Create tensor of embeddings from word iterator."""
    tensor = torch.LongTensor(len(iterator))
    token = 0
    for item in iterator:
        tensor[token] = dictionary.word2idx[item] if item in dictionary.word2idx else dictionary.word2idx['<unk>']
        token += 1
    return tensor
