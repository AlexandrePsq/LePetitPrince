from nltk.tokenize import sent_tokenize 
from nltk.tokenize import word_tokenize
import os
import re
import inflect
from tqdm import tqdm


special_words = {
    'english': {
        'grown-ups': 'grownups',
        'grown-up': 'grownup',
        'hasn\'t': 'hasnt',
        'hasn‘t': 'hasnt'
    },
    'french': {

    }
}


def tokenize(path, language, vocab=None, path_like=True, train=False):
    print('Tokenizing...')
    if path_like:
        assert os.path.exists(path)
        path = open(path, 'r', encoding='utf8').read()

    print('Preprocessing...')
    text = preprocess(path, special_words, language)
    print('Preprocessed.')

    if train:
        text = text.split('.')
        iterator = [unk_transform_sentence(item.split(), vocab) for item in tqdm(text)]
    else:
        text = text.replace('.', ' ')
        iterator = [unk_transform(item, vocab) for item in tqdm(text.split())]
    print('Tokenized.')
    return iterator


def unk_transform_sentence(sentence, vocab=None):
    result = []
    for word in sentence:
        result.append(unk_transform(word, vocab))
    return result

def unk_transform(word, vocab=None):
    if word == 'unk':
        return '<unk>'
    elif not vocab:
        return word.lower()
    elif word in vocab.idx2word:
        return word.lower()
    else:
        return '<unk>'


def preprocess(text, special_words, language):
    text = text.replace('\n', '')
    text = text.replace('<unk>', 'unk')
    for word in special_words[language].keys():
        text = text.replace(word, special_words[language][word])
    transf = inflect.engine()
    numbers = re.findall('\d+', text)
    for number in numbers:
        text = text.replace(number, transf.number_to_words(number))
    eos = ['...', '!', '?', '..']
    for item in eos:
        text = text.replace(item, '.')
    punctuation = ['\'', ',', ';', ':', '/', '-', '"', '‘', '’', '(', ')', '{', '}', '[', ']', '`', '“', '”', '—']
    ## tokenize without punctuation ###
    for item in punctuation:
        text = text.replace(item, ' ')
    return text