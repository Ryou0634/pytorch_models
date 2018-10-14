from torch_models.models.transformer import *


from my_utils import Dictionary

n_unique = 10

src_dict = Dictionary(['<EOS>'])
tgt_dict = Dictionary(['<BOS>', '<EOS>'])
for n in range(n_unique):
    src_dict.add_word(str(n))
    tgt_dict.add_word(str(n))

from my_utils.toy_data import invert_seq
train = invert_seq(1, n_unique=n_unique)

import torch
from my_utils import DataLoader
from torch_models.utils import seq2seq
def numericalize(dataset, src_dict, tgt_dict):
    numericalized = [([src_dict(s) for s in src], [tgt_dict(t) for t in tgt]) for src, tgt in dataset]
    return numericalized

# device = 'cuda:0'
device = torch.device('cpu')
trans_func = seq2seq(device)
train_loader = DataLoader(numericalize(train, src_dict, tgt_dict), batch_size=1, trans_func=trans_func)


embed_size = 24
n_head = 4
train_model = Transformer(size=embed_size, n_head=n_head, src_vocab_size=len(src_dict), tgt_vocab_size=len(tgt_dict),
                    src_EOS=src_dict('<EOS>'), tgt_BOS=tgt_dict('<BOS>'), tgt_EOS=tgt_dict('<EOS>'),
                    dropout=0, n_layers=2)
from copy import deepcopy
test_model = deepcopy(train_model)

inputs, targets = next(train_loader)

def test_decoder_layer():
    encoded, enc_seq_lens = model.encode(inputs_EOS)
    train_decoder = train_model.decoder
    test_decoder = test_model.decoder

    # enc_outputs, enc_seq_lens を用意
