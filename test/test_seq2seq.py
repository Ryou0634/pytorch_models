
import torch
from torch_models.models import Seq2Seq

model = Seq2Seq(embed_size=1, hidden_size=5, src_vocab_size=2, tgt_vocab_size=3,
                src_EOS=0, tgt_BOS=0, tgt_EOS=2, num_layers=1, bidirectional=False,
                rnn='LSTM')

def test_encode():
    inputs = [torch.LongTensor([1, 1]), torch.LongTensor([1]), torch.LongTensor([1, 1])]
    encoded = model.encode(inputs)
    outputs = encoded['outputs']
    lengths = encoded['lengths']
    hiddens = encoded['hiddens']
    hiddens = hiddens[0]

    assert (outputs[0] == outputs[2]).all()
    assert (hiddens[:, 0] == hiddens[:, 2]).all()
    assert not (outputs[0] == outputs[1]).all()
    assert not (hiddens[:, 0] == hiddens[:, 1]).all()
    assert (torch.tensor([3, 2, 3]) == lengths).all()

def test_append_EOS():
    inputs = [torch.LongTensor([1]), torch.LongTensor([1, 1, 1])]
    inputs_EOS = model._append_EOS(inputs)
    expected = [torch.LongTensor([1, 0]), torch.LongTensor([1, 1, 1, 0])]
    for exp, seq in zip(expected, inputs_EOS):
        assert (exp == seq).all()

def test_apend_BOS():
    inputs = [torch.LongTensor([1]), torch.LongTensor([1, 1, 1])]
    inputs_BOS = model._append_BOS(inputs)
    expected = [torch.LongTensor([0, 1]), torch.LongTensor([0, 1, 1, 1])]
    for exp, seq in zip(expected, inputs_BOS):
        assert (exp == seq).all()

def test_append_EOS_flatten():
    inputs = [torch.LongTensor([1]), torch.LongTensor([1, 1, 1])]
    outputs = model._append_EOS_flatten(inputs)
    expected = torch.LongTensor([1, 2, 1, 1, 1, 2])
    assert (expected == outputs).all()

def test_flatten_and_unpad():
    # (batch, max_length, embed_dim)
    decoded = torch.tensor([[[1, 1, 1],
                             [2, 2, 2]],
                            [[3, 3, 3],
                             [4, 4, 4]],
                            [[5, 5, 5],
                             [6, 6, 6]]])
    lengths = [1, 2, 1]
    outputs = model._flatten_and_unpad(decoded, lengths)
    expected = torch.tensor([[1, 1, 1],
                             [3, 3, 3],
                             [4, 4, 4],
                             [5, 5, 5]])
    for exp, seq in zip(expected, outputs):
        assert (exp == seq).all()

import torch
from my_utils import Dictionary, DataLoader
from my_utils.toy_data import invert_seq
from torch_models.utils import seq2seq
def numericalize(dataset, src_dict, tgt_dict):
    numericalized = [([src_dict(s) for s in src], [tgt_dict(t) for t in tgt]) for src, tgt in dataset]
    return numericalized

def get_toy_data_loader():
    n_unique = 10
    src_dict = Dictionary(['<EOS>'])
    tgt_dict = Dictionary(['<BOS>', '<EOS>'])
    for n in range(n_unique):
        src_dict.add_word(str(n))
        tgt_dict.add_word(str(n))
    train = invert_seq(5000, n_unique=n_unique)
    test = invert_seq(100, n_unique=n_unique)

    device = torch.device('cpu')
    trans_func = seq2seq(device)
    train_loader = DataLoader(numericalize(train, src_dict, tgt_dict), batch_size=64, trans_func=trans_func)
    test_loader = DataLoader(numericalize(test, src_dict, tgt_dict), batch_size=50, trans_func=trans_func)
    return train_loader, test_loader, src_dict, tgt_dict

from torch_models import AttnSeq2Seq, Seq2Seq
from my_utils import Trainer, EvaluatorSeq, EvaluatorLoss
from torch.optim import Adam, SGD
import numpy as np

import pytest
def test_seq2seq():
    train_loader, test_loader, src_dict, tgt_dict = get_toy_data_loader()
    embed_size=64
    dropout = 0
    model = Seq2Seq(embed_size=embed_size, hidden_size=embed_size, src_vocab_size=len(src_dict), tgt_vocab_size=len(tgt_dict),
                        src_EOS=src_dict('<EOS>'), tgt_BOS=tgt_dict('<BOS>'), tgt_EOS=tgt_dict('<EOS>'),
                        num_layers=1, bidirectional=True, dropout=dropout, rnn='LSTM')

    optimizer = Adam(model.parameters())

    trainer = Trainer(model, train_loader)
    trainer.train_epoch(optimizer, max_epoch=5,
                  evaluator=None, score_monitor=None)
    test_evaluator = EvaluatorSeq(model, test_loader, measure='sent_BLEU')
    assert 0.8 < test_evaluator.evaluate()

    # check if greedy_predict and predict(beam-search) give the same outputs.
    gre_predicted = []
    predicted = []
    model.beam_width = 1
    for inputs, target in test_loader:
        gre_predicted += model.greedy_predict(inputs)
        predicted += model.beam_search(inputs)
    for gre, pre in zip(gre_predicted, predicted):
        assert (np.array(gre) == np.array(pre)).all()

@pytest.mark.parametrize(
    "rnn, attention, fuse_query", [
        ('LSTM', 'dot', 'add'),
        ('GRU', 'bilinear', 'linear'),
    ]
)
def test_attnseq2seq(rnn, attention, fuse_query):
    train_loader, test_loader, src_dict, tgt_dict = get_toy_data_loader()
    embed_size=64
    dropout = 0
    model = AttnSeq2Seq(embed_size=embed_size, hidden_size=embed_size, src_vocab_size=len(src_dict), tgt_vocab_size=len(tgt_dict),
                        src_EOS=src_dict('<EOS>'), tgt_BOS=tgt_dict('<BOS>'), tgt_EOS=tgt_dict('<EOS>'),
                        num_layers=1, bidirectional=True, dropout=dropout, rnn=rnn,
                        attention=attention, fuse_query=fuse_query)

    optimizer = Adam(model.parameters())

    trainer = Trainer(model, train_loader)
    trainer.train_epoch(optimizer, max_epoch=5,
                  evaluator=None, score_monitor=None)
    test_evaluator = EvaluatorSeq(model, test_loader, measure='BLEU')
    assert 0.8 < test_evaluator.evaluate()

@pytest.mark.parametrize(
    "rnn, attention, fuse_query", [
        ('LSTM', 'dot', 'add'),
        ('GRU', 'bilinear', 'linear'),
    ]
)
def test_decode(rnn, attention, fuse_query):
    train_loader, test_loader, src_dict, tgt_dict = get_toy_data_loader()
    embed_size=64
    dropout = 0
    model = AttnSeq2Seq(embed_size=embed_size, hidden_size=embed_size, src_vocab_size=len(src_dict), tgt_vocab_size=len(tgt_dict),
                        src_EOS=src_dict('<EOS>'), tgt_BOS=tgt_dict('<BOS>'), tgt_EOS=tgt_dict('<EOS>'),
                        num_layers=1, bidirectional=True, dropout=dropout, rnn=rnn,
                        attention=attention, fuse_query=fuse_query)
    inputs, targets = next(train_loader)
    encoded = model.encode(inputs)
    decoded1 = model.decode_input_feeding(targets, encoded)
    decoded2 = model.decode(targets, encoded)
    assert (decoded1 == decoded2).all()




