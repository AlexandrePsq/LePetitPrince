import sys
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)


import torch
import torch.nn as nn
import math
import pandas as pd
import numpy as np
import time
from .data import Corpus
from tqdm import tqdm
from utilities.settings import Params
from .model import RNNModel
from .utils import get_batch, repackage_hidden, batchify, save, load

params = Params()



###############################################################################
# Evaluating code
###############################################################################

def evaluate(model, criterion, ntokens, data_source, eval_batch_size):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in tqdm(range(0, data_source.size(0) - 1, params.pref.bptt)):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / (len(data_source) - 1)



###############################################################################
# Training code
###############################################################################

def forward(model, train_data, corpus, criterion, epoch, lr, bsz=params.pref.bsz):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(params.pref.bsz)
    for batch, i in tqdm(enumerate(range(0, train_data.size(0) - 1, params.pref.bptt))):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), params.pref.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if batch % params.pref.log_interval == 0 and batch > 0:
            cur_loss = total_loss / params.pref.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // params.pref.bptt, lr,
                elapsed * 1000 / params.pref.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def train(model, data, data_name, language, eval_batch_size=params.pref.eval_batch_size, bsz=params.pref.bsz, epochs=params.pref.epochs):
    torch.manual_seed(params.pref.seed) # setting seed for reproductibility
    device = torch.device("cuda" if params.cuda else "cpu")
    corpus = Corpus(data, language)
    train_data = batchify(corpus.train, bsz, device)
    val_data = batchify(corpus.valid, eval_batch_size, device)
    test_data = batchify(corpus.test, bsz, device)

    # Build the model
    ntokens = len(corpus.dictionary)
    param = model.param
    param['ntoken'] = ntokens
    model = RNNModel(**param)
    #model.encoder.num_embeddings = ntokens
    #model.decoder.out_features = ntokens
    model.vocab = corpus.dictionary
    model = model.to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()

    # Loop over epochs.
    best_val_loss = None
    lr = params.pref.lr

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        print('Entering training...')
        for epoch in tqdm(range(1, epochs+1)):
            epoch_start_time = time.time()
            forward(model, train_data, corpus, criterion, epoch, lr)
            val_loss = evaluate(model, criterion, ntokens, val_data, eval_batch_size)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                            val_loss, math.exp(val_loss)))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                save(model, data_name, language)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    print('loading best saved model...')
    model = load(model, data_name, language)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.rnn.flatten_parameters()

    # Run on test data.
    print('evaluation...')
    test_loss = evaluate(model, criterion, ntokens, test_data, eval_batch_size)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)