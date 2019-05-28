################################################################
# General Language Model including :
#   - an embedding layer
#   - LSTM layers: (nlayers, nhidden)
#
################################################################
import sys
import os
from data import Corpus, Dictionary
from tokenizer import tokenize, batchify

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)


import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from ....utilities.settings import Params

params = Params()


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        # hack the forward function to send an extra argument containing the model parameters
	    # self.rnn.forward = lambda input, hidden: lstm.forward(model.rnn, input, hidden)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.vocab = None
    
    def init_vocab(self, path):
        self.vocab = Dictionary(path)

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

    def generate(self, path, language):
        # self.init_vocab(vocab_path)
        # hack the forward function to send an extra argument containing the model parameters
        # self.rnn.forward = lambda input, hidden: lstm.forward(model.rnn, input, hidden)
        columns_activations = ['raw-features-activations-{}'.format(i) for i in range(self.nhid * self.nlayers)]
        activations = []
        surprisals = []
        iterator = tokenize(path, language, self.vocab)
        last_item = params.eos_separator
        out = None
        hidden = None
        for item in iterator:
            activation, surprisal,(out, hidden) = self.extract_activations(item, 
                                                                            last_item=last_item, 
                                                                            surprisal=params.surprisal, 
                                                                            out=out, 
                                                                            hidden=hidden)
            last_item = item
            activations.append(activation[0][0])
            surprisals.append(surprisal)
        activations_df = pd.DataFrame(np.vstack(activations), columns=columns_activations)
        surprisals_df = pd.DataFrame(np.vstack(surprisals), columns=['surprisal'])
        return pd.concat([activations_df, surprisals_df], axis = 1)
    
    def extract_activations(self, item, last_item, surprisal=False, out=None, hidden=None):
        if last_item == params.eos_separator:
            hidden = self.init_hidden(1)
            inp = torch.autograd.Variable(torch.LongTensor([[self.vocab.word2idx[params.eos_separator]]]))
            if params.cuda:
                inp = inp.cuda()
            out, hidden = self(inp, hidden)
        surprisal = out[0, 0, self.vocab.word2idx[item]].item()
        inp = torch.autograd.Variable(torch.LongTensor([[self.vocab.word2idx[item]]]))
        if params.cuda:
            inp = inp.cuda()
        out, hidden = self(inp, hidden)
        activation = hidden[0].data.view(1,1,-1).cpu().numpy()
        return activation, surprisal, (out, hidden)


def load(path):
    return torch.load(path)
    
def save(model, path):
    return torch.save(model, path)


# python LSTM/prepare_LSTM_features/extract_activations/extract_activations.py \
# LSTM/LSTM_training/word_language_model/English_model_wikitext_LSTM_seed_1_nhid_2_emsize_2_nlayers_5_dropout_0.5_batchsize_64_lr_10_epochs_10_0.pt \
#  -v LSTM/LSTM_training/word_language_model/data/wiki_kristina/vocab.txt \
#  -o LSTM/prepare_LSTM_features/extract_activations/english_all.pkl \
#  -i LSTM/LSTM_training/word_language_model/data/wikitext-2/train.txt --use-unk --get-representations lstm
# 
# 
# python main.py  --model LSTM --data ./data/test --seed 1 --nhid 2 --emsize 2 --nlayers 5 --dropout 0.5 --batch_size 64 --lr 10 --epochs 10 --save ./English_model_wikitext_LSTM_seed_1_nhid_2_emsize_2_nlayers_5_dropout_0.5_batchsize_64_lr_10_epochs_10_0.pt


# model = module.load()
# model.generate(path)