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
import h5py
from utils import match_tokenized_to_untokenized, extract_activations_from_tokenized



parameters = {'base':{'LAYER_COUNT':12, 'FEATURE_COUNT':768},
                'large':{'LAYER_COUNT':24, 'FEATURE_COUNT':1024}
            }



class BERT(object):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, bert_model, language, name, loi):
        super(BERT, self).__init__()
        # Load pre-trained model tokenizer (vocabulary)
        # Crucially, do not do basic tokenization; PTB is tokenized. Just do wordpiece tokenization.
        if bert_model not in ['base', 'large']:
            raise ValueError("BERT model must be base or large")
        self.model = BertModel.from_pretrained('bert-{}-cased'.format(bert_model))
        self.tokenizer = BertTokenizer.from_pretrained('bert-{}-cased'.format(bert_model))
        self.language = language
        self.LAYER_COUNT = parameters[bert_model]['LAYER_COUNT']
        self.FEATURE_COUNT = parameters[bert_model]['FEATURE_COUNT']
        self.name = name
        self.generation = self.name.split('-')[2]
        self.loi = np.array(loi) if loi else np.arange(parameters[bert_model]['LAYER_COUNT']) # loi: layers of interest

    def __name__(self):
        return self.name


    def generate(self, path, language, textgrid):
        """ Input text should have one sentence per line, where each word and every 
        symbol is separated from the following by a space. No <eos> token should be included,
        as they are automatically integrated during tokenization.
        """
        activations = []
        self.model.eval()
        if self.generation == 'bucket':
            # Here, we give as input the text line by line.
            for line in open(path):
                line = line.strip() # Remove trailing characters

                line = '[CLS] ' + line + ' [SEP]'
                tokenized_text = self.tokenizer.wordpiece_tokenizer.tokenize(line)
                indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
                segment_ids = [1 for x in tokenized_text]
                mapping = match_tokenized_to_untokenized(tokenized_text, line)

                # Convert inputs to PyTorch tensors
                tokens_tensor = torch.tensor([indexed_tokens])
                segments_tensors = torch.tensor([segment_ids])

                with torch.no_grad():
                    encoded_layers, _ = self.model(tokens_tensor, segments_tensors) # dimension = layer_count * len(tokenized_text) * feature_count
                    # filtration
                    encoded_layers = encoded_layers[self.loi, :, :]
                    activations += extract_activations_from_tokenized(encoded_layers.numpy(), mapping, tokenized_text)
        elif self.generation == 'sequential':
            # Here we give as input the sentence up to the actual word, incrementing by one at each step.
            for line in open(path):
                for index in range(1, len(line.split())):
                    tmp_line = " ".join(line.split()[:index])
                    tmp_line = tmp_line.strip() # Remove trailing characters

                    tmp_line = '[CLS] ' + tmp_line + ' [SEP]'
                    tokenized_text = self.tokenizer.wordpiece_tokenizer.tokenize(tmp_line)
                    indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
                    segment_ids = [1 for x in tokenized_text]

                    # Convert inputs to PyTorch tensors
                    tokens_tensor = torch.tensor([indexed_tokens])
                    segments_tensors = torch.tensor([segment_ids])

                    with torch.no_grad():
                        encoded_layers, _ = self.model(tokens_tensor, segments_tensors) # dimension = layer_count * len(tokenized_text) * feature_count
                        # filtration
                        encoded_layers = encoded_layers[self.loi, :, :]
                        activations.append(np.mean(encoded_layers.numpy(), axis=1).reshape(1,-1))
        result = pd.DataFrame(np.vstack(activations), columns=['layer-{}-{}'.format(layer, index) for layer in self.loi for index in range(self.FEATURE_COUNT)])
        return result