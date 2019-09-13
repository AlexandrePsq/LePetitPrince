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

parameters = {'base':{'LAYER_COUNT':12, 'FEATURE_COUNT':768},
                'large':{'LAYER_COUNT':24, 'FEATURE_COUNT':1024}
            }

class BERT(object):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, bert_model, tokenizer, language):
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

    def __name__(self):
        pass
        #return '_'.join([function.__name__ for function in self.functions])

    def generate(self, path, language, textgrid):
        self.model.eval()
        for line in open(path):
            line = line.strip() # Remove trailing characters
            line = '[CLS] ' + line + ' [SEP]'
            tokenized_text = self.tokenizer.wordpiece_tokenizer.tokenize(line)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            segment_ids = [1 for x in tokenized_text]
        
            # Convert inputs to PyTorch tensors
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segment_ids])
        
            with torch.no_grad():
                encoded_layers, _ = self.model(tokens_tensor, segments_tensors)
        # iterator = tokenize(path, language) 
        iterator = list(textgrid['word']) # we suppose the textgrid dataframe (=csv file with onsets and offsets issue from original textgrid) has been created thanks to the tokennize function
        dataframes = [pd.DataFrame(function(iterator, language), columns=[function.__name__]) for function in self.functions]
        result = pd.concat([df for df in dataframes], axis = 1)
        return result


  



#with h5py.File(args.output_path, 'w') as fout:
#  
#    dset = fout.create_dataset(str(index), (LAYER_COUNT, len(tokenized_text), FEATURE_COUNT))
#    dset[:,:,:] = np.vstack([np.array(x) for x in encoded_layers])