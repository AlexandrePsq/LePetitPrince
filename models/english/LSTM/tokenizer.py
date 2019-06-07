from nltk.tokenize import sent_tokenize 
from nltk.tokenize import word_tokenize
import os


def tokenize(path, language, vocab=None, path_like=True):
    if path_like:
        assert os.path.exists(path)
        path = open(path, 'r', encoding='utf8').read()
    text = path.replace('\n', '')
    text = text.replace('<unk>', 'unk')
    tokenized = sent_tokenize(text, language=language)
    tokenized = [word_tokenize(sentence, language=language) + ['<eos>'] for sentence in tokenized]
    iterator = [unk_transform(item, vocab).lower() for sublist in tokenized for item in sublist]
    return iterator

def unk_transform(word, vocab=None):
    if word == 'unk':
        return '<unk>'
    elif not vocab:
        return word
    elif word in vocab.idx2word:
        return word
    else:
        return '<unk>'
