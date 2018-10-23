import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from torch_models.models import SeqEncoderBase

class TransformerEmbedding(SeqEncoderBase):
    def __init__(self, embed_size, vocab_size):
        super().__init__(embed_size, vocab_size)

    def forward(self, seqs):
        # padding
        padded_seqs, seq_lens = self._pad_seqs(seqs)
        # get embedding
        embeds = self.embedding(padded_seqs) * math.sqrt(self.embed_size) # (batch, max_length, embed_size)
        return embeds, seq_lens

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        # Compute the positional encodings once in log space.
        div_term = torch.exp(torch.arange(0.0, embed_size, 2) *
                             -(math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, inputs):
        inputs = inputs + Variable(self.pe[:inputs.size(1)], # (batch, seq_len, embed_size)
                                   requires_grad=False)
        return self.dropout(inputs)

from torch_models.models import DotAttn
class MultiHeadedAttention(nn.Module):
    def __init__(self, input_size, n_head, subsequent_mask=False, dropout=0.1):
        super().__init__()
        assert input_size % n_head == 0

        self.d_k = input_size//n_head
        self.n_head = n_head

        self.Q_linear = nn.Linear(input_size, input_size) # virtually (input_size, self.d_k) * n_head
        self.K_linear = nn.Linear(input_size, input_size)
        self.V_linear = nn.Linear(input_size, input_size)
        self.out_linear = nn.Linear(input_size, input_size)
        self.attention = DotAttn(scaled=True, subsequent_mask=subsequent_mask)
        self.attn_weights = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, queries, keys, values, key_lens=None, query_lens=None):
        '''
        Parameters
        -----------
        queries : torch.tensor (batch, n_queries, dim1)
        keys : torch.tensor (batch, n_keys , dim1) (this is also regarded as values)
        values : torch.tensor (batch, n_keys, dim_value)
        query_lens : torch.LongTensor
            Contains the number of queries of each batch.
        key_lens : torch.LongTensor
            Contains the number of keys of each batch.
        '''

        batchsize = len(queries)
        # transformation
        # (batchsize, n_queries, self.d_k)*self.n_head -> (batchsize*self.n_head, n_queries, self.d_k)
        Qs, Ks, Vs = [torch.cat(l(x).split(self.d_k, dim=2)) for x, l
                      in zip([queries, keys, values], [self.Q_linear, self.K_linear, self.V_linear])]
        # attention
        multi_head_vecs, self.attn_weights = self.attention(Qs, Ks, Vs,
                        query_lens=query_lens.repeat(self.n_head), key_lens=key_lens.repeat(self.n_head))
        multi_head_vecs = torch.cat(multi_head_vecs.split(batchsize), dim=2) # (batchsize, n_queries, self.n_head*self.d_k)
        # output
        return self.out_linear(multi_head_vecs) # (batchsize, n_queries, input_size)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, size, n_head, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadedAttention(size, n_head, dropout=dropout)
        self.fc = PositionwiseFeedForward(size, size*4, dropout=dropout)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(size) for _ in range(2)])

    def forward(self, inputs, seq_lens):
        attn = self.attention(inputs, inputs, inputs, seq_lens, seq_lens) # (batchsize, max_seq_len, size)
        h1 = self.layer_norms[0](attn + inputs)
        feeded = self.fc(h1)
        outputs = self.layer_norms[1](feeded+h1)
        return outputs # (batchsize, max_seq_len, size)

class TransformerEncoder(nn.Module):
    def __init__(self, size, n_head, n_vocab, n_layers=6, dropout=0.1):
        super().__init__()
        self.embedding = TransformerEmbedding(size, n_vocab)
        self.pe = PositionalEncoding(size, dropout=dropout)
        self.layers = nn.ModuleList([EncoderLayer(size, n_head, dropout) for _ in range(n_layers)])

    def forward(self, inputs):
        embeds, enc_seq_lens = self.embedding(inputs) # embeds is padded. (batchsize, max_seq_len, size)
        x = self.pe(embeds)
        for layer in self.layers:
            x = layer(x, enc_seq_lens)
        return x, enc_seq_lens

class DecoderLayer(nn.Module):
    def __init__(self, size, n_head, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadedAttention(size, n_head, subsequent_mask=True, dropout=dropout)
        self.src_tgt_attention = MultiHeadedAttention(size, n_head, dropout=dropout)
        self.fc = PositionwiseFeedForward(size, size*4, dropout=dropout)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(size) for _ in range(3)])


    def forward(self, inputs, seq_lens, enc_outputs, enc_seq_lens):
        # inputs : (batchsize, max_seq_len, size)
        self_attn = self.self_attention(queries=inputs, keys=inputs, values=inputs,
                                        query_lens=seq_lens, key_lens=seq_lens)
        h1 = self.layer_norms[0](self_attn + inputs)

        src_tgt_attn = self.src_tgt_attention(queries=h1, keys=enc_outputs, values=enc_outputs,
                                              query_lens=seq_lens, key_lens=enc_seq_lens)
        h2 = self.layer_norms[1](src_tgt_attn+h1)
        feeded = self.fc(h2)
        outputs = self.layer_norms[2](feeded+h2)
        return outputs # (batchsize, max_seq_len, size)

class TransformerDecoder(nn.Module):
    def __init__(self, size, n_head, n_vocab, n_layers=6, dropout=0.1):
        super().__init__()
        self.embedding = TransformerEmbedding(size, n_vocab)
        self.pe = PositionalEncoding(size, dropout=dropout)
        self.layers = nn.ModuleList([DecoderLayer(size, n_head, dropout) for _ in range(n_layers)])

    def forward(self, inputs, enc_outputs, enc_seq_lens):
        embeds, seq_lens = self.embedding(inputs)  # embeds is padded. (batchsize, max_seq_len, size)
        x = self.pe(embeds)
        for layer in self.layers:
            x = layer(x, seq_lens, enc_outputs, enc_seq_lens)
        return x, seq_lens

from torch_models.models import Seq2SeqBase, MLP
class Transformer(Seq2SeqBase):
    def __init__(self, size, n_head, src_vocab_size, tgt_vocab_size,
                 src_EOS, tgt_BOS, tgt_EOS, n_layers=1, dropout=0.1):
        super().__init__(src_EOS, tgt_BOS, tgt_EOS)

        self.encoder = TransformerEncoder(size=size, n_head=n_head, n_vocab=src_vocab_size,
                                          n_layers=n_layers, dropout=dropout)
        self.decoder = TransformerDecoder(size=size, n_head=n_head, n_vocab=tgt_vocab_size,
                                          n_layers=n_layers, dropout=dropout)
        self.generator = MLP(dims=[size, tgt_vocab_size])
        self.initialize()

    def initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, inputs):
        inputs_EOS = self._append_EOS(inputs)
        enc_outputs, enc_seq_lens = self.encoder(inputs_EOS) # (batchsize, max_seq_len, size)
        return {'outputs': enc_outputs,
                'lengths': enc_seq_lens}

    def decode(self, inputs, encoded):
        enc_outputs = encoded['outputs']
        enc_seq_lens = encoded['lengths']
        decoded, dec_seq_lens = self.decoder(inputs, enc_outputs, enc_seq_lens) # (batchsize, max_seq_len, size)
        dec_outputs = self._flatten_and_unpad(decoded, dec_seq_lens) # (n_tokens, size)
        return {'outputs': dec_outputs}

    # used in beam_search()
    def _get_last_hidden(self, decoded, batchsize, i):
        return decoded['outputs'].view(batchsize, i+1, -1)[:, -1] # extract the last one

    # used in greedy_predict()
    def _update(self, input_tokens, output_tokens, encoded, decoded):
        input_tokens = torch.cat((input_tokens, output_tokens.unsqueeze(1)), dim=1)
        return input_tokens, encoded

    def _update_beam(self, input_tokens, next_tokens, encoded, decoded, parent_hypos, beam_searcher):
        input_tokens = torch.cat((input_tokens[parent_hypos], next_tokens.unsqueeze(1)), dim=1)
        encoded['outputs'] = encoded['outputs'][:beam_searcher.width]
        encoded['lengths'] = encoded['lengths'][:beam_searcher.width]
        return input_tokens, encoded
