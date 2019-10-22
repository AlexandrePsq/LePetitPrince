import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)

import argparse
import warnings
warnings.simplefilter(action='ignore')

# from nltk.tokenize import sent_tokenize 
# from nltk.tokenize import word_tokenize
from tokenizer import tokenize
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import glob
import re
import inflect

########################## PARSING ##########################

parser = argparse.ArgumentParser(description="Create the onsets for a given tokenizer relatively to an initial set of onsets")

parser.add_argument('--model_name', type=str, help='Name of the model to consider')
parser.add_argument('--language', type=str, help='Language of the text being tokenized')
parser.add_argument('--onsets', help='path to initial onsets folder')
parser.add_argument('--text', help='path to the folder containing the text to tokenize and for which we need the onsets')
parser.add_argument('--save', type=str, help='path to the folder in which to save the new onsets files')

args = parser.parse_args()




########################## PREPROCESSING ##########################

special_words = {'grown-ups': 'grownups',
                    'hasn\'t': 'hasnt',
                    'hasn’t': 'hasnt',
                    'grown-up':'grownup'
}


def preprocess(text, special_words):
    for word in special_words.keys():
        text = text.replace(word, special_words[word])
    transf = inflect.engine()
    numbers = re.findall('\d+', text)
    for number in numbers:
        text = text.replace(number, transf.number_to_words(number))
    return text







########################## MAIN ##########################


if __name__ == '__main__':
    text_files = sorted(glob.glob(os.path.join(args.text, 'text_{}_run*.txt'.format(args.language))))
    onsets_files = sorted(glob.glob(os.path.join(args.onsets, 'text_{}_onsets-offsets_run*.csv'.format(args.language))))
    
    try:
        assert len(onsets_files)==len(text_files)
        try:
            assert len(onsets_files)==9
        except AssertionError:
            print('You are working with {} runs instead of 9. Be careful to the data you are using and check the code/utilities/settings.py file to specify the new number of runs'.format(len(onsets_files)))
        
        for run in tqdm(range(len(onsets_files))):
            result = []
            ref_df = pd.read_csv(onsets_files[run])
            ref_words = list(ref_df.word)
            text = open(text_files[run], 'r').read().lower()
            text = preprocess(text, special_words)
            tmp_text = None
            for index in range(len(ref_words)):
                new_index = text.find(ref_words[index])
                tmp_text = text[: new_index]
                text = text[new_index + len(ref_words[index]):]
                # Extrapolating onset-offset values
                words = tokenize(tmp_text, args.language, path_like=False)
                onsets = np.linspace(ref_df.onsets.iloc[max(0, index-1)], ref_df.onsets.iloc[index], len(words))
                offsets = np.linspace(ref_df.offsets.iloc[max(0, index-1)], ref_df.offsets.iloc[index], len(words))
                result += list(zip(onsets, offsets, words))
                result.append(((ref_df['onsets'].iloc[index], ref_df['offsets'].iloc[index], ref_df['word'].iloc[index])))
            df = pd.DataFrame(result, columns=['onsets', 'offsets', 'word'])
            saving_path = os.path.join(args.save, 'text_{}_{}_onsets-offsets_run{}.csv'.format(args.language, args.model_name, run + 1))
            df.to_csv(saving_path)
            
    except AssertionError:
        print('You do not have the same number of onsets files and text-to-tokenize files ...')
