import argparse
import torch
from data import Corpus


parser = argparse.ArgumentParser(description='Use a given model to predict the future words --> check model quality')

parser.add_argument('--data', type=str, default='../../../../data/text/english/english/lstm_training',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='../../../../derivatives/fMRI/models/english/LSTM_embedding-size_200_nhid_100_nlayers_3_dropout_0.1_wiki_kristina_english.pt')
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--cuda', action='store_true', help='Use CUDA')
parser.add_argument('--temperature', type=float, default=1.0, help='temperature - higher will increase diversity')
parser.add_argument('--save', type=str, default='../../../../derivatives/fMRI/models/english/generated.txt')
parser.add_argument('--words', type=int, default=1000, help='number of words to generate')
parser.add_argument('--log_interval', type=int, default=100, help='reporting interval')


args = parser.parse_args()



torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

if args.temperature < 1e-3 :
    parser.error('--temperature has to be greater or equal to 1e-3')

with open(args.model, 'rb') as f:
    model = torch.load(f).to(device)
model.eval()

corpus = Corpus(args.data)
ntokens = len(Corpus.dictionary)
hidden = model.init_hidden(1)
input = torch.randint(ntokens, (1,1), dtype=torch.long).to(device)

with open(args.save, 'w') as outf:
    with torch.no_grad(): # no tracking history
        for i in range(args.words):
            output, hidden = model(input, hidden)
            word_weights = output.squeeze().div(args.temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            input.fill_(word_idx)
            word = corpus.dictionary.idx2word[word_idx]

            outf.write(word + ('\n' if i % 20 == 19 else ' '))

            if i % args.log_interval == 0:
                print('| Generated {}/{} words'.format(i, args.words))