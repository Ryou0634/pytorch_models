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
        self.embedding = nn.Embedding(vocab_size+1, embed_size, padding_idx=vocab_size)

class BoV(SeqEncoderBase):
    def __init__(self, embed_size, vocab_size):
        super().__init__(embed_size, vocab_size)
        self.output_size = embed_size

    def _get_embeds(self, inputs):
        # flatten inputs, get embeddings, then split them back
        seq_lengths = [len(seq) for seq in inputs]
        cat_seqs = torch.cat(inputs)
        embed_seqs = self.embedding(cat_seqs).split(seq_lengths)
        return embed_seqs

    def forward(self, inputs):
        embed_seqs = self._get_embeds(inputs)
        averaged = [torch.mean(embed_seqs[i], dim=0) for i in range(len(embed_seqs))]
        return torch.stack(averaged)

class LSTMEncoder(SeqEncoderBase):
    def __init__(self, embed_size, hidden_size, vocab_size, bidirectional=False, num_layers=1):
        super().__init__(embed_size, vocab_size)
        self.rnn = nn.LSTM(embed_size, hidden_size,
                        bidirectional=bidirectional, num_layers=num_layers)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 1+bidirectional
        self.output_size = embed_size*self.num_directions

    def get_packed_embeds(self, seqs):
        # sorting
        original_idx, sorted_seqs = self._sort_seqs(seqs)
        # padding
        padded_seqs, original_lengths = self._pad_seqs(sorted_seqs)
        # get embedding
        embeds = self.embedding(padded_seqs) # (batch, max_length, embed_size)
        # packing
        packed_embeds = pack_padded_sequence(embeds, original_lengths, batch_first=True)
        return packed_embeds, original_idx

    def _sort_seqs(self, seqs):
        idx_and_seqs = list(enumerate(seqs))
        idx_and_seqs.sort(key=lambda x: len(x[1]), reverse=True)
        original_idx, sorted_seqs = zip(*idx_and_seqs)
        return original_idx, sorted_seqs

    def _pad_seqs(self, sorted_seqs):
        lengths = [len(seq) for seq in sorted_seqs]
        max_length = lengths[0]
        padded_seqs = [torch.cat((sorted_seqs[i],
                                 self.embedding.padding_idx*torch.ones(max_length - lengths[i], dtype=torch.int64)))
                      for i in range(len(sorted_seqs))]
        padded_seqs = torch.stack(padded_seqs, dim=0)
        return padded_seqs, lengths

    def reorder_batch(self, tensor, original_idx):
        _, idxs = torch.sort(torch.tensor(original_idx))
        return tensor[idxs]

    def forward(self, inputs, init_hiddens=None):
        packed_embeds, original_idx = self.get_packed_embeds(inputs)
        if init_hiddens is not None:
            init_state = (init_hiddens[:, original_idx], torch.zeros(init_hiddens.shape, device=init_hiddens.device))
            outputs, (hiddens, _) = self.rnn(packed_embeds, init_state)
        else:
            outputs, (hiddens, _) = self.rnn(packed_embeds)
        tensors, lengths = pad_packed_sequence(outputs, batch_first=True)

        for tensor, length in zip(tensors, lengths):
            tensor[length:] = float('-inf')

        # reorder_batch
        _, idxs = torch.sort(torch.tensor(original_idx))
        tensors = tensors[idxs]
        hiddens = hiddens[:, idxs]
        return tensors, hiddens # (batch, max_length, embed_dim), (num_layers * num_directions, batch, hidden_size)

class LSTMLastHidden(LSTMEncoder):
    def __init__(self, embed_size, vocab_size, bidirectional=False, num_layers=1):
        super().__init__(embed_size, embed_size, vocab_size, bidirectional, num_layers)

    def forward(self, inputs):
        packed_embeds, original_idx = self.get_packed_embeds(inputs)
        _, (hiddens, _) = self.rnn(packed_embeds) # (num_layers * num_directions, batch, hidden_size)
        hiddens = hiddens.view(self.num_layers, self.num_directions, -1, self.embed_size)
        hidden = torch.cat([tensor for tensor in hiddens[-1]], dim=1) # only use hidden from the last layer and concat along the dim of num_direction
        hidden = self.reorder_batch(hidden, original_idx)
        return hidden

class LSTMMaxPool(LSTMEncoder):
    def __init__(self, embed_size, vocab_size, bidirectional=False, num_layers=1):
        super().__init__(embed_size, embed_size, vocab_size, bidirectional, num_layers)

    def forward(self, inputs):
        packed_embeds, original_idx = self.get_packed_embeds(inputs)

        outputs, _ = self.rnn(packed_embeds)
        tensors, lengths = pad_packed_sequence(outputs, batch_first=True)
        for tensor, length in zip(tensors, lengths):
            tensor[length:] = float('-inf')
        pooled, _ = torch.max(tensors, dim=1)
        pooled = self.reorder_batch(pooled, original_idx)
        return pooled
