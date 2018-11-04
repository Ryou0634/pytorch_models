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

    def _pad_seqs(self, inputs):
        seq_lens = torch.LongTensor([len(seq) for seq in inputs])
        padded_seqs = self.embedding.padding_idx*inputs[0].new_ones((len(inputs), seq_lens.max()))
        for i, (seq, seq_len) in enumerate(zip(inputs, seq_lens)):
            padded_seqs[i, :seq_len] = seq
        return padded_seqs, seq_lens


class RNNEncoder(SeqEncoderBase):
    def __init__(self, embed_size, hidden_size, vocab_size, bidirectional=None, num_layers=1,
                 dropout=0, rnn='RNN', feed_size=0):
        super().__init__(embed_size, vocab_size)
        rnn_unit = eval('nn.'+rnn)

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

        self.rnn = rnn_unit(embed_size+feed_size, hidden_size,
                        bidirectional=self.bidirectional, num_layers=num_layers, dropout=dropout)
        self.hidden_size = hidden_size
        self.feed_size = feed_size
        self.num_layers = num_layers
        self.num_directions = 1+self.bidirectional

    def pad_and_sort_inputs(self, inputs):
        # padding
        padded_seqs, seq_lens = self._pad_seqs(inputs) # (batch, max_len)
        # sorting
        seq_lens, perm_idx = seq_lens.sort(descending=True)
        padded_seqs = padded_seqs[perm_idx]
        return padded_seqs, seq_lens, perm_idx

    def get_packed_embeds(self, inputs):
        padded_seqs, seq_lens, perm_idx = self.pad_and_sort_inputs(inputs)
        # get embedding
        embeds = self.embedding(padded_seqs) # (batch, max_len, embed_size)
        # packing
        packed_embeds = pack_padded_sequence(embeds, seq_lens, batch_first=True)
        return packed_embeds, perm_idx

    def _reorder_hiddens(self, hiddens, perm_idx):
        if isinstance(self.rnn, nn.LSTM):
            hiddens, cells = hiddens
            return (hiddens[:, perm_idx], cells[:, perm_idx])
        else:
            return hiddens[:, perm_idx]

    def _copy_hiddens(self, hiddens, n_copy):
        ''' Copy hiddens along the batch dimension. '''
        if isinstance(self.rnn, nn.LSTM):
            hiddens, cells = hiddens
            return (hiddens.repeat(1, n_copy, 1), cells.repeat(1, n_copy, 1))
        else:
            return hiddens.repeat(1, n_copy, 1)

    def _add_along_direction(self, tensors, hiddens):
        ''' Used when merging two hiddens from a bidirectional RNN. '''
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
        '''
        Compute batched outputs through all time steps.
        Inputs will be padded, and so is the outputs. You can unpad the outputs with lengths.

        Parameters :
        inputs : List[torch.LongTensor (seq_len, )]
            A batch of tensors containing indices.
        init_state : torch.tensor or Tuple[torch.tensor] (for LSTM)
            Initial hidden state.

        Returns :
        tensors : torch.tensor (batch, max(lengths), hidden_size)
        lengths : torch.LongTensor
        hiddens : torch.tensor or Tuple[torch.tensor] (for LSTM)
        '''
        packed_embeds, perm_idx = self.get_packed_embeds(inputs)
        if init_state is not None:
            init_state = self._reorder_hiddens(init_state, perm_idx)
        outputs, hiddens = self.rnn(packed_embeds, init_state)
        tensors, lengths = pad_packed_sequence(outputs, batch_first=True)

        if self.bidirectional and self.bidir_type == 'add':
            tensors, hiddens = self._add_along_direction(tensors, hiddens)

        # _reorder_batch
        _, unperm_idx = perm_idx.sort()
        lengths = lengths[unperm_idx]
        tensors = tensors[unperm_idx]                        # (batch, seq_len, output_size)
        hiddens = self._reorder_hiddens(hiddens, unperm_idx) # (num_layers * num_directions, batch, hidden_size)
        return (tensors, lengths), hiddens

    def forward_step(self, inputs, init_state=None, feed_vec=None):
        '''
        Compute batched outputs for a single step.

        Parameters :
        inputs : torch.LongTensor (batch, )
            A batch of tensors containing indices.
        init_state : torch.tensor or Tuple[torch.tensor] (for LSTM)
            Initial hidden state.
        feed_vec : torch.tensor (batch, feed_size)
            Additional inputs concatenated to embeddings.

        Returns :
        tensors : torch.tensor (batch, 1, hidden_size)
        hiddens : torch.tensor or Tuple[torch.tensor] (for LSTM)
        '''
        assert (self.feed_size == 0 and feed_vec is None) or (self.feed_size > 0 and feed_vec is not None)
        embeds = self.embedding(inputs) # (batch, input_size)
        if feed_vec is not None:
            embeds = torch.cat([embeds, feed_vec], dim=1) # (batch, input_size+feed_size)
        tensors, hiddens = self.rnn(embeds.unsqueeze(0), init_state)
        # (1, batch, num_directions * hidden_size), (num_layers * num_directions, batch, hidden_size)
        tensors = tensors.permute(1, 0, 2)
        if self.bidirectional and self.bidir_type == 'add':
            tensors, hiddens = self._add_along_direction(tensors, hiddens)
        return tensors, hiddens