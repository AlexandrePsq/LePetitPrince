from nltk.tokenize import sent_tokenize 
from nltk.tokenize import word_tokenize


def tokenize(path, language, vocab):
    raw_text = open(path, 'r').read()
    text = raw_text.replace('\n', '')
    text = text.replace('<unk>', 'unk')
    tokenized = sent_tokenize(text, language=language)
    tokenized = [word_tokenize(sentence, 'english') + ['<eos>'] for sentence in tokenized]
    iterator = [unk_transform(item, vocab).lower() for sublist in tokenized for item in sublist]
    return iterator

def unk_transform(word, vocab):
    if word == 'unk':
        return '<unk>'
    elif word in vocab.idx2word:
        return word
    else:
        return '<unk>'

def batchify(data, bsz, device):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)