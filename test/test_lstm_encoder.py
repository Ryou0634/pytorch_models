import torch
from torch_models.models import LSTMEncoder

model = LSTMEncoder(embed_size=10, hidden_size=10, vocab_size=1)

seqs = [torch.LongTensor([0, 0, 0, 0]),
        torch.LongTensor([0, 0, 0, 0, 0]),
        torch.LongTensor([0, 0])]

def test_sort_seqs():
    original_idx, sorted_seqs = model._sort_seqs(seqs)

    expected = (torch.LongTensor([0, 0, 0, 0, 0]),
                torch.LongTensor([0, 0, 0, 0]),
                torch.LongTensor([0, 0]))
    for exp, seq in zip(expected, sorted_seqs):
        assert (exp == seq).all()
    assert (1, 0, 2) == original_idx

def test_pad_seqs():
    _, sorted_seqs = model._sort_seqs(seqs)
    padded_seqs, lengths = model._pad_seqs(sorted_seqs)
    expected = (torch.LongTensor([0, 0, 0, 0, 0]),
                torch.LongTensor([0, 0, 0, 0, 1]),
                torch.LongTensor([0, 0, 1, 1, 1]))
    for exp, seq in zip(expected, padded_seqs):
        assert (exp == seq).all()
    assert [5, 4, 2] == lengths


def test_reorder_batch():
    tensor = torch.eye(3, dtype=torch.int64)
    original_idx = [1, 2, 0]
    reoredered = model.reorder_batch(tensor, original_idx)
    expected = (torch.LongTensor([0, 0, 1]),
                torch.LongTensor([1, 0, 0]),
                torch.LongTensor([0, 1, 0]))
    for exp, seq in zip(expected, reoredered):
        assert (exp == seq).all()
