import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SeqEncoderBase(nn.Module):
    '''
    Sequence encoder takes sequences as input, transform them into embeddings,
    and then outputs encoded fixed-size representation.
    '''
    def __init__(self, embed_size, vocab_size):
        super().__init__()
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_size)

    def _get_embeds(self, inputs):
        # flatten inputs, get embeddings, then split them back
        seq_lengths = [len(seq) for seq in inputs]
        cat_seqs = torch.cat(inputs)
        embed_seqs = self.embedding(cat_seqs).split(seq_lengths)
        return embed_seqs

class BoV(SeqEncoderBase):
    def __init__(self, embed_size, vocab_size):
        super().__init__(embed_size, vocab_size)

    def forward(self, inputs):
        embed_seqs = self._get_embeds(inputs)
        averaged = [torch.mean(embed_seqs[i], dim=0) for i in range(len(embed_seqs))]
        return torch.stack(averaged)

class LSTMEncoderBase(SeqEncoderBase):
    def __init__(self, embed_size, vocab_size, bidirectional, num_layers):
        super().__init__(embed_size, vocab_size)


        self.rnn = nn.LSTM(embed_size, embed_size,
                        bidirectional=bidirectional, num_layers=num_layers)

        self.num_layers = num_layers
        self.num_directions = 1+bidirectional
        self.output_size = embed_size*self.num_directions

    def _pack_embeds(self, embed_seqs):
        # sort embed_seqs according to the length
        idx_embed_seqs = list(enumerate(embed_seqs))
        idx_embed_seqs.sort(key=lambda x: len(x[1]), reverse=True)
        original_idx, embed_seqs = zip(*idx_embed_seqs)
        batch_sizes = [len(seq) for seq in embed_seqs]
        # put the seqs into a padded tensor
        padded = torch.zeros((len(embed_seqs), batch_sizes[0], self.embed_size), device=embed_seqs[0][0].device)
        for i, embed_seq in enumerate(embed_seqs):
            for j, embed in enumerate(embed_seq):
                padded[i][j] = embed
        packed_seqs = pack_padded_sequence(padded, batch_sizes, batch_first=True)
        return packed_seqs, original_idx

    def _reorder_batch(self, tensor, original_idx):
        _, idxs = torch.sort(torch.tensor(original_idx))
        return tensor[idxs]

class LSTMLastHidden(LSTMEncoderBase):
    def __init__(self, embed_size, vocab_size, bidirectional=False, num_layers=1):
        super().__init__(embed_size, vocab_size, bidirectional, num_layers)


    def forward(self, inputs):
        embed_seqs = self._get_embeds(inputs)
        packed_seqs, original_idx = self._pack_embeds(embed_seqs)

        _, (hidden, _) = self.rnn(packed_seqs)
        hidden = hidden.view(self.num_layers, self.num_directions, -1, self.embed_size)
        hidden = torch.cat([tensor for tensor in hidden[-1]], dim=1)
        # put it back to the right order
        hidden = self._reorder_batch(hidden, original_idx)
        return hidden

class LSTMMaxPool(LSTMEncoderBase):
    def __init__(self, embed_size, vocab_size, bidirectional=False, num_layers=1):
        super().__init__(embed_size, vocab_size, bidirectional, num_layers)


    def forward(self, inputs):
        embed_seqs = self._get_embeds(inputs)
        packed_seqs, original_idx = self._pack_embeds(embed_seqs)

        outputs, _ = self.rnn(packed_seqs)
        tensors, lengths = pad_packed_sequence(outputs, batch_first=True)
        for tensor, length in zip(tensors, lengths):
            tensor[length:] = -math.inf
        pooled, _ = torch.max(tensors, dim=1)
        pooled = self._reorder_batch(pooled, original_idx)
        return pooled
