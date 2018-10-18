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

class RNNEncoder(SeqEncoderBase):
    def __init__(self, embed_size, hidden_size, vocab_size, bidirectional=None, num_layers=1,
                 dropout=0, rnn='rnn'):
        super().__init__(embed_size, vocab_size)

        if rnn is 'rnn':
            rnn_unit = nn.RNN
        elif rnn is 'lstm':
            rnn_unit = nn.LSTM
        elif rnn is 'gru':
            rnn_unit = nn.GRU
        else:
            raise Exception("rnn must be ['rnn', 'lstm', 'gru']")

        if bidirectional is None:
            self.bidirectional = False
            self.output_size = hidden_size
        elif bidirectional == 'add':
            self.bidirectional = True
            self.bidir_type = 'add'
            self.output_size = hidden_size
        elif bidirectional == 'cat':
            self.bidirectional = True
            self.bidir_type = 'cat'
            self.output_size = 2*hidden_size
        else:
            raise Exception("bidirectional must be [None, 'add', 'cat']")

        self.rnn = rnn_unit(embed_size, hidden_size,
                        bidirectional=self.bidirectional, num_layers=num_layers, dropout=dropout)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 1+self.bidirectional

    def _pad_seqs(self, inputs, seq_lens):
        padded_seqs = self.embedding.padding_idx*inputs[0].new_ones((len(inputs), seq_lens.max()))
        for i, (seq, seq_len) in enumerate(zip(inputs, seq_lens)):
            padded_seqs[i, :seq_len] = seq
        return padded_seqs

    def get_packed_embeds(self, inputs):
        seq_lens = torch.LongTensor([len(seq) for seq in inputs])
        # padding
        padded_seqs = self._pad_seqs(inputs, seq_lens) # (batch, max_len)
        # sorting
        seq_lens, perm_idx = seq_lens.sort(descending=True)
        padded_seqs = padded_seqs[perm_idx]
        # get embedding
        embeds = self.embedding(padded_seqs) # (batch, max_len, embed_size)
        # packing
        packed_embeds = pack_padded_sequence(embeds, seq_lens, batch_first=True)
        _, unperm_idx = perm_idx.sort()
        return packed_embeds, unperm_idx

    def _reorder_hiddens(self, hiddens, unperm_idx):
        if isinstance(self.rnn, nn.LSTM):
            hiddens, cells = hiddens
            hiddens = hiddens[:, unperm_idx]
            cells = cells[:, unperm_idx]
            return (hiddens, cells)
        else:
            return hiddens[:, unperm_idx]

    def _add_along_direction(self, tensors, hiddens):
        batchsize = tensors.size(0)
        tensors = tensors.view(batchsize, -1, self.num_directions, self.hidden_size).sum(dim=2) # (batch, seq_len, hidden_size)
        if isinstance(self.rnn, nn.LSTM):
            hiddens, cells = hiddens
            hiddens = hiddens.view(self.num_layers, self.num_directions, -1, self.hidden_size).sum(dim=1)
            cells = cells.view(self.num_layers, self.num_directions, -1, self.hidden_size).sum(dim=1)
            hiddens = (hiddens, cells)
        else:
            hiddens = hiddens.view(self.num_layers, self.num_directions, -1, self.hidden_size).sum(dim=1)
        return tensors, hiddens

    def forward(self, inputs, init_state=None):
        packed_embeds, unperm_idx = self.get_packed_embeds(inputs)
        if init_state is not None:
            outputs, hiddens = self.rnn(packed_embeds, init_state)
        else:
            outputs, hiddens = self.rnn(packed_embeds)
        tensors, lengths = pad_packed_sequence(outputs, batch_first=True)

        if self.bidirectional and self.bidir_type == 'add':
            tensors, hiddens = self._add_along_direction(tensors, hiddens)

        # _reorder_batch
        lengths = lengths[unperm_idx]
        tensors = tensors[unperm_idx]                        # (batch, seq_len, output_size)
        hiddens = self._reorder_hiddens(hiddens, unperm_idx) # (num_layers * num_directions, batch, hidden_size)
        return (tensors, lengths), hiddens

class RNNLastHidden(RNNEncoder):
    def __init__(self, embed_size, hidden_size, vocab_size, bidirectional='cat', num_layers=1, dropout=0, rnn='lstm'):
        super().__init__(embed_size=embed_size, hidden_size=hidden_size, vocab_size=vocab_size,
                         bidirectional=bidirectional, num_layers=num_layers, dropout=dropout, rnn=rnn)

    def forward(self, inputs):
        packed_embeds, unperm_idx = self.get_packed_embeds(inputs)
        _, hiddens = self.rnn(packed_embeds) # (num_layers * num_directions, batch, hidden_size)
        if isinstance(self.rnn, nn.LSTM):
            hiddens = hiddens[0]
        hiddens = hiddens.view(self.num_layers, self.num_directions, -1, self.hidden_size)
        if not self.bidirectional:
            hidden = hiddens[-1].squeeze(0)
        if self.bidirectional and self.bidir_type == 'add':
            hidden = hiddens.sum(dim=1) # (num_layers, batch, hidden_size)
            hidden = hidden[-1] # (batch, hidden_size)
        if self.bidirectional and self.bidir_type == 'cat':
            hidden = torch.cat([tensor for tensor in hiddens[-1]], dim=1) # (batch, output_size)
        return hidden[unperm_idx] # (batch, output_size)

class RNNMaxPool(RNNEncoder):
    def __init__(self, embed_size, hidden_size, vocab_size, bidirectional='cat', num_layers=1, dropout=0, rnn='lstm'):
        super().__init__(embed_size, hidden_size, vocab_size, bidirectional, num_layers, dropout, rnn)

    def forward(self, inputs):
        packed_embeds, unperm_idx = self.get_packed_embeds(inputs)

        outputs, _ = self.rnn(packed_embeds) # (batch, seq_len, num_directions * hidden_size)
        tensors, lengths = pad_packed_sequence(outputs, batch_first=True)
        if self.bidirectional and self.bidir_type == 'add':
            tensors = tensors.view(len(inputs), -1, self.num_directions, self.hidden_size).sum(dim=2) # (batch, seq_len, hidden_size)
        for tensor, length in zip(tensors, lengths):
            tensor[length:] = float('-inf')
        pooled, _ = torch.max(tensors, dim=1)
        return pooled[unperm_idx] # (batch, output_size)
