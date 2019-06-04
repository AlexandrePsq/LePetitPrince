import sys
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)

import torch
import torch.nn.functional as F
from utilities.settings import Params, Paths

params = Params()
paths = Paths()

###############################################################################
# Utilities
###############################################################################


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    seq_len = min(params.pref.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def batchify(data, bsz, device):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def save(model, data_name, language):
    path = '_'.join([model.__name__(), data_name, language]) + '.pt'
    path = os.path.join(paths.path2derivatives, 'fMRI/models', language, path)
    with open(path, 'wb') as f:
        torch.save(model, f)


def load(model, data_name, language):
    path = '_'.join([model.__name__(), data_name, language]) + '.pt'
    path = os.path.join(paths.path2derivatives, 'fMRI/models', language, path)
    assert os.path.exists(path)
    with open(path, 'rb') as f:
        return torch.load(f)



###############################################################################
# Extracting advanced features
###############################################################################

def LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None): 
    hx, cx = hidden
    gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

    ingate, forgetgate, cy_tilde, outgate = gates.chunk(4, 1) #dim modified from 1 to 2

    ingate = F.sigmoid(ingate)
    forgetgate = F.sigmoid(forgetgate)
    cy_tilde = F.tanh(cy_tilde)
    outgate = F.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cy_tilde)
    hy = outgate * F.tanh(cy)

    return {'hidden': hy, 'cell': cy, 'in': ingate, 'forget': forgetgate, 'out': outgate, 'c_tilde': cy_tilde}


def apply_mask(hidden_l, mask):
    if type(hidden_l) == torch.autograd.Variable:
        return hidden_l * mask
    else:
        return tuple(h * mask for h in hidden_l)


def forward(self, input, hidden, param, mask=None):
    weight = self.all_weights
    dropout = param['dropout']
    # saves the gate values into the rnn object
    last_gates = []

    hidden = list(zip(*hidden))

    for l in range(param['nlayers']):
        hidden_l = hidden[l]
        if mask and l in mask:
            hidden_l = apply_mask(hidden_l, mask[l])
        # we assume there is just one token in the input
        gates = LSTMCell(input[0], hidden_l, *weight[l])
        hy = (gates['hidden'], gates['cell'])
        if mask and l in mask:
            hy = apply_mask(hy, mask[l])

        last_gates.append(gates)
        input = hy[0]

        if dropout != 0 and l < param['nlayers'] - 1:
            input = F.dropout(input, p=dropout, training=False, inplace=False)

    self.gates =  {key: torch.cat([last_gates[i][key].unsqueeze(0) for i in range(param['nlayers'])], 0) for key in ['in', 'forget', 'out', 'c_tilde', 'hidden', 'cell']}
    self.hidden = {key: torch.cat([last_gates[i][key].unsqueeze(0) for i in range(param['nlayers'])], 0) for key in ['hidden', 'cell']}
    # we restore the right dimensionality
    input = input.unsqueeze(0)
    return input, (self.hidden['hidden'], self.hidden['cell'])