import torch
from torch_models.models import Seq2Seq

model = Seq2Seq(embed_size=1, hidden_size=5, src_vocab_size=2, tgt_vocab_size=3,
                src_EOS=0, tgt_BOS=0, tgt_EOS=2, num_layers=1, bidirectional=False,
                rnn='lstm')

def test_encode():
    inputs = [torch.LongTensor([1, 1]), torch.LongTensor([1]), torch.LongTensor([1, 1])]
    (outputs, lengths), hiddens = model.encode(inputs)
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
