################################################################
# BERT Language Model
################################################################
import sys
import os

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if root not in sys.path:
    sys.path.append(root)


import pandas as pd
import numpy as np
import torch
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM, WordpieceTokenizer
from . import utils
from .tokenizer import tokenize 


parameters = {'base':{'LAYER_COUNT':12, 'FEATURE_COUNT':768},
                'large':{'LAYER_COUNT':24, 'FEATURE_COUNT':1024}
            }



class BERT(object):
    """Container module for BERT."""

    def __init__(self, bert_model, language, name, loi):
        super(BERT, self).__init__()
        # Load pre-trained model tokenizer (vocabulary)
        # Crucially, do not do basic tokenization; PTB is tokenized. Just do wordpiece tokenization.
        if bert_model not in ['base', 'large']:
            raise ValueError("BERT model must be base or large")
        self.model = BertModel.from_pretrained('bert-{}-cased'.format(bert_model), output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained('bert-{}-cased'.format(bert_model))
        
        self.language = language
        self.LAYER_COUNT = parameters[bert_model]['LAYER_COUNT']
        self.FEATURE_COUNT = parameters[bert_model]['FEATURE_COUNT']
        self.name = name
        self.generation = self.name.split('-')[2].strip()
        self.loi = np.array(loi) if loi else np.arange(1 + parameters[bert_model]['LAYER_COUNT']) # loi: layers of interest

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
        if self.generation == 'bucket':
            # Here, we give as input the text line by line.
            for line in iterator:
                line = line.strip() # Remove trailing characters

                line = '[CLS] ' + line + ' [SEP]'
                tokenized_text = self.tokenizer.wordpiece_tokenizer.tokenize(line)
                indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
                segment_ids = [1 for x in tokenized_text]
                mapping = utils.match_tokenized_to_untokenized(tokenized_text, line)

                # Convert inputs to PyTorch tensors
                tokens_tensor = torch.tensor([indexed_tokens])
                segments_tensors = torch.tensor([segment_ids])

                with torch.no_grad():
                    encoded_layers = self.model(tokens_tensor, segments_tensors) # last_hidden_state, pooled_last_hidden_states, all_hidden_states
                    # filtration
                    encoded_layers = np.vstack(encoded_layers[2][1:]) # retrieve all the hidden states (dimension = layer_count * len(tokenized_text) * feature_count)
                    encoded_layers = encoded_layers[self.loi, :, :]
                    activations += utils.extract_activations_from_tokenized(encoded_layers, mapping)
        elif self.generation == 'sequential':
            # Here we give as input the sentence up to the actual word, incrementing by one at each step.
            for line in iterator:
                for index in range(1, len(line.split())):
                    tmp_line = " ".join(line.split()[:index])
                    tmp_line = tmp_line.strip() # Remove trailing characters

                    tmp_line = '[CLS] ' + tmp_line + ' [SEP]'
                    tokenized_text = self.tokenizer.wordpiece_tokenizer.tokenize(tmp_line)
                    indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
                    segment_ids = [1 for x in tokenized_text]
                    mapping = utils.match_tokenized_to_untokenized(tokenized_text, line)

                    # Convert inputs to PyTorch tensors
                    tokens_tensor = torch.tensor([indexed_tokens])
                    segments_tensors = torch.tensor([segment_ids])

                    with torch.no_grad():
                        encoded_layers = self.model(tokens_tensor, segments_tensors) # dimension = layer_count * len(tokenized_text) * feature_count
                        # filtration
                        encoded_layers = np.vstack(encoded_layers[2])
                        encoded_layers = encoded_layers[self.loi, :, :]
                        activations.append(utils.extract_activations_from_tokenized(encoded_layers, mapping)[-1])
        result = pd.DataFrame(np.vstack(activations), columns=['layer-{}-{}'.format(layer, index) for layer in self.loi for index in range(self.FEATURE_COUNT)])
        return result