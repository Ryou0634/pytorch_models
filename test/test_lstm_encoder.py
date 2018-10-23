import torch
from torch_models.models import RNNEncoder

model = RNNEncoder(embed_size=10, hidden_size=10, vocab_size=1)

seqs = [torch.LongTensor([0, 0, 0, 0]),
        torch.LongTensor([0, 0, 0, 0, 0]),
        torch.LongTensor([0, 0])]

def test_pad_seqs():
    padded_seqs, seq_lens = model._pad_seqs(seqs)
    expected = (torch.LongTensor([0, 0, 0, 0, 1]),
                torch.LongTensor([0, 0, 0, 0, 0]),
                torch.LongTensor([0, 0, 1, 1, 1]))
    for exp, seq in zip(expected, padded_seqs):
        assert (exp == seq).all()

    expected_len = torch.LongTensor([4, 5, 2])
    assert (expected_len == seq_lens).all()
