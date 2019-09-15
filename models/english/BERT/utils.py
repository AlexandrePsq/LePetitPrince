from collections import namedtuple, defaultdict
import os
import numpy as np



def match_tokenized_to_untokenized(tokenized_sent, untokenized_sent):
    '''Aligns tokenized and untokenized sentence given subwords "##" prefixed
    Assuming that each subword token that does not start a new word is prefixed
    by two hashes, "##", computes an alignment between the un-subword-tokenized
    and subword-tokenized sentences.
    Args:
        tokenized_sent: a list of strings describing a subword-tokenized sentence
        untokenized_sent: a list of strings describing a sentence, no subword tok.
    Returns:
        A dictionary of type {int: list(int)} mapping each untokenized sentence
        index to a list of subword-tokenized sentence indices
    '''
    mapping = defaultdict(list)
    untokenized_sent_index = 0
    tokenized_sent_index = 1
    while (untokenized_sent_index < len(untokenized_sent) and tokenized_sent_index < len(tokenized_sent)):
        while (tokenized_sent_index + 1 < len(tokenized_sent) and tokenized_sent[tokenized_sent_index + 1].startswith('##')):
            mapping[untokenized_sent_index].append(tokenized_sent_index)
            tokenized_sent_index += 1
        mapping[untokenized_sent_index].append(tokenized_sent_index)
        untokenized_sent_index += 1
        tokenized_sent_index += 1
    return mapping


def extract_activations_from_tokenized(activation, mapping):
    """Take the average activations of the tokens related to a given word."""
    nb_words = activation.shape[1]
    new_activations = []
    for word_index in range(nb_words):
        word_activation = []
        word_activation.append([activation[0,index, :] for index in mapping[word_index]])
        word_activation = np.vstack(word_activation)
        new_activations.append(np.mean(word_activation, axis=0).reshape(1,-1))
    return new_activations