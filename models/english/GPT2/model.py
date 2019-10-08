################################################################
# GPT2 Language Model
################################################################
import sys
import os

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if root not in sys.path:
    sys.path.append(root)


import pandas as pd
import numpy as np
import torch
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM, WordpieceTokenizer, GPT2Tokenizer, GPT2Model
from . import utils
from .tokenizer import tokenize 


parameters = {'small':{'LAYER_COUNT':12+1, 'FEATURE_COUNT':768},
                'medium':{'LAYER_COUNT':24+1, 'FEATURE_COUNT':1024}
            }



class GPT2(object):
    """Container module for GPT2."""

    def __init__(self, gpt2_model, language, name, loi, cuda=False):
        super(GPT2, self).__init__()
        # Load pre-trained model tokenizer (vocabulary)
        # Crucially, do not do basic tokenization; PTB is tokenized. Just do wordpiece tokenization.
        if gpt2_model not in ['small', 'medium']:
            raise ValueError("GPT2 model must be small or medium")
        self.model = GPT2Model.from_pretrained('gpt2{}'.format('' if gpt2_model=='small' else '-medium'))
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2{}'.format('' if gpt2_model=='small' else '-medium'))
        self.model.config.output_hidden_states = True
        self.model.save_pretrained('./')
        self.tokenizer.save_pretrained('./')
        self.model = GPT2Model.from_pretrained('./')
        self.tokenizer = GPT2Tokenizer.from_pretrained('./')

        self.language = language
        self.LAYER_COUNT = parameters[gpt2_model]['LAYER_COUNT']
        self.FEATURE_COUNT = parameters[gpt2_model]['FEATURE_COUNT']
        self.name = name
        self.generation = self.name.split('-')[2].strip()
        self.loi = np.array(loi) if loi else np.arange(parameters[gpt2_model]['LAYER_COUNT']) # loi: layers of interest
        self.cuda = cuda

    def __name__(self):
        return self.name


    def generate(self, path, language, textgrid):
        """ Input text should have one sentence per line, where each word and every 
        symbol is separated from the following by a space. No <eos> token should be included,
        as they are automatically integrated during tokenization.
        """
        activations = []
        self.model.eval()
        iterator = tokenize(path, language, path_like=True, train=False)
        if self.cuda:
            self.model.to('cuda')
        for line in iterator:
            line = line.strip() # Remove trailing characters

            tokenized_text = self.tokenizer.tokenize(line)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            mapping = utils.match_tokenized_to_untokenized(tokenized_text, line)

            # Convert inputs to PyTorch tensors
            tokens_tensor = torch.tensor([indexed_tokens]).to('cuda') if self.cuda else torch.tensor([indexed_tokens])

            with torch.no_grad():
                encoded_layers = self.model(tokens_tensor) # last_hidden_state, pooled_last_hidden_states, all_hidden_states
                # filtration
                if self.cuda:
                    encoded_layers = encoded_layers.to('cpu')
                encoded_layers = np.vstack(encoded_layers[2][1:]) # retrieve all the hidden states (dimension = layer_count * len(tokenized_text) * feature_count)
                encoded_layers = encoded_layers[self.loi, :, :]
                activations += utils.extract_activations_from_tokenized(encoded_layers, mapping)
       
        result = pd.DataFrame(np.vstack(activations), columns=['layer-{}-{}'.format(layer, index) for layer in self.loi for index in range(self.FEATURE_COUNT)])
        return result

