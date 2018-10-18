import torch
from torch_models.models import RNNEncoder

model = RNNEncoder(embed_size=10, hidden_size=10, vocab_size=1)

seqs = [torch.LongTensor([0, 0, 0, 0]),
        torch.LongTensor([0, 0, 0, 0, 0]),
        torch.LongTensor([0, 0])]

def test_pad_seqs():
    seq_lens = torch.LongTensor([len(seq) for seq in seqs])
    padded_seqs = model._pad_seqs(seqs, seq_lens)
    expected = (torch.LongTensor([0, 0, 0, 0, 1]),
                torch.LongTensor([0, 0, 0, 0, 0]),
                torch.LongTensor([0, 0, 1, 1, 1]))
    for exp, seq in zip(expected, padded_seqs):
        assert (exp == seq).all()
