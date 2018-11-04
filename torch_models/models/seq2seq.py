from torch_models.models import MLP, RNNEncoder, BeamSearcher
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Seq2SeqBase(nn.Module):
    def __init__(self, src_EOS, tgt_BOS, tgt_EOS, beam_width=1):
        super().__init__()
        self.src_EOS = src_EOS
        self.tgt_BOS = tgt_BOS
        self.tgt_EOS = tgt_EOS
        self.beam_width = beam_width

    def fit(self, inputs, targets, optimizer):
        if optimizer:
            self.train()
        else:
            self.eval()
        self.zero_grad()
        # encoding
        encoded = self.encode(inputs) # (num_layers, batch, hidden_size)
        # decoding
        bos_targets = self._append_BOS(targets)
        decoded = self.decode(bos_targets, encoded)
        # predicting
        targets_eos = self._append_EOS_flatten(targets)
        loss_item = self.generator.fit(decoded['outputs'], targets_eos, optimizer)
        return loss_item

    def predict(self, inputs, max_len=50):
        if self.beam_width == 1:
            return self.greedy_predict(inputs, max_len)
        else:
            return self.beam_search(inputs, max_len) # List[List[int]]

    def beam_search(self, inputs, max_len=50):
        """
        During beam-search, computation is done with a batch_size of beam_width.
        """
        self.eval()
        predicted = []
        with torch.no_grad():
            batchsize = len(inputs)
            # encoding
            encoded_batch = self.encode(inputs)
            for n in range(batchsize):
                encoded = self._extract_repeat(encoded_batch, n, self.beam_width)
                input_tokens = inputs[0].new_tensor([[self.tgt_BOS] for _ in range(self.beam_width)]) # (beam_width, 1)
                beam_searcher = BeamSearcher(self.beam_width, self.tgt_EOS)
                for i in range(max_len):
                    decoded  = self.decode(input_tokens, encoded) # (beam_width, hidden_size)
                    last_hidden = self._get_last_hidden(decoded, beam_searcher.width, i)
                    log_p = F.log_softmax(self.generator.forward(last_hidden), dim=1) # (beam_width, tgt_vocab_size)
                    next_tokens, parent_hypos = beam_searcher.step(i, log_p)
                    if beam_searcher.width == 0: break
                    # preparing for the next iteration
                    input_tokens, encoded = self._update_beam(input_tokens, next_tokens, encoded, decoded, parent_hypos, beam_searcher)
                top_seqs = beam_searcher.hypos + beam_searcher.end_hypos
                predicted.append(max(top_seqs, key=lambda x: x['score'])['seq'])
        return predicted

    def _update_beam(self, input_tokens, next_tokens, encoded, decoded, parent_hypos, beam_searcher):
       raise NotImplementedError()

    def _extract_repeat(self, encoded_batch, n, beam_width):
        outputs = encoded_batch['outputs'][n].repeat(beam_width, 1, 1)
        lengths = encoded_batch['lengths'][n].repeat(beam_width)
        encoded = {'outputs': outputs, 'lengths': lengths}
        if 'hiddens' in encoded_batch.keys():
            hiddens = self.encoder._reorder_hiddens(encoded_batch['hiddens'], [n])
            hiddens = self.encoder._copy_hiddens(hiddens, beam_width)
            encoded.update({'hiddens': hiddens})
        return encoded


    def greedy_predict(self, inputs, max_len=50):
        self.eval()
        predicted = [] # contains selected idxs in each time step
        with torch.no_grad():
            # encoding
            encoded = self.encode(inputs) # encoded['outputs']: (batchsize, max_seq_len, size)
            batchsize = len(inputs)
            input_tokens = inputs[0].new_tensor([self.tgt_BOS for _ in range(batchsize)]).view(-1, 1) # (batchsize, 1)
            end_flags = inputs[0].new_zeros(batchsize)
            for i in range(max_len):
                decoded  = self.decode(input_tokens, encoded)
                last_hidden = self._get_last_hidden(decoded, batchsize, i) # (batchsize, hidden_size)
                output_tokens = self.generator.predict(last_hidden) # (batchsize,)
                predicted.append(output_tokens)
                end_flags.masked_fill_(output_tokens.eq(self.tgt_EOS), 1) # set 1 in end_flags if EOS
                if end_flags.sum() == batchsize: break # end if all flags are 1
                input_tokens, encoded = self._update(input_tokens, output_tokens, encoded, decoded)
        predicted = torch.stack(predicted, dim=1)
        return self._remove_EOS(predicted) # List[torch.tensor]

    def _get_last_hidden(self, decoded, batchsize, i):
        raise NotImplementedError()

    def _update(self, input_tokens, output_tokens, encoded, decoded):
        raise NotImplementedError()

    def encode(self, inputs):
        raise NotImplementedError()

    def decode(self, inputs, encoded):
        raise NotImplementedError()

    def _remove_EOS(self, predicted):
        outputs = []
        for seq in predicted:
            if self.tgt_EOS in seq:
                EOS_idx = list(seq).index(self.tgt_EOS)
                outputs.append(seq[:EOS_idx])
            else:
                outputs.append(seq)
        return outputs

    def _append_EOS(self, inputs):
        inputs_eos = [torch.cat((inp, inp.new_tensor([self.src_EOS]))) for inp in inputs]
        return inputs_eos

    def _append_BOS(self, targets):
        bos_targets = [torch.cat((target.new_tensor([self.tgt_BOS]), target)) for target in targets]
        return bos_targets

    def _append_EOS_flatten(self, targets):
        EOS_targets = [torch.cat((target, target.new_tensor([self.tgt_EOS]))) for target in targets]
        return torch.cat(EOS_targets)

    def _flatten_and_unpad(self, decoded, lengths):
        # decoded: (batch, max_length, embed_dim)
        unpadded = [batch[:l] for batch, l in zip(decoded, lengths)]
        flattened = torch.cat(unpadded, dim=0)
        return flattened # (batch_sizes, embed_dim)

class Seq2Seq(Seq2SeqBase):
    def __init__(self, embed_size, hidden_size, src_vocab_size, tgt_vocab_size,
                 src_EOS, tgt_BOS, tgt_EOS, num_layers=1, bidirectional=False, dropout=0, rnn='LSTM',
                 init_w=0.1, beam_width=1, input_feeding=False):
        if bidirectional:
            bidir_type = 'add'
        else:
            bidir_type = None
        super().__init__(src_EOS, tgt_BOS, tgt_EOS, beam_width)
        self.encoder = RNNEncoder(embed_size, hidden_size, src_vocab_size,
                                  bidirectional=bidir_type, num_layers=num_layers, dropout=dropout, rnn=rnn)
        self.hidden_size = hidden_size

        feed_size = hidden_size if input_feeding else 0
        self.input_feeding = input_feeding
        self.decoder = RNNEncoder(embed_size, self.hidden_size, tgt_vocab_size,
                                   bidirectional=None, num_layers=num_layers, dropout=dropout, rnn=rnn,
                                   feed_size=feed_size)
        self.generator = MLP(dims=[self.hidden_size, tgt_vocab_size], dropout=dropout)

        self.initialize(init_w)

    def initialize(self, init_w):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.uniform_(p, -init_w, init_w)

    def encode(self, inputs):
        inputs_eos = self._append_EOS(inputs)
        (enc_outputs, lengths), hiddens = self.encoder(inputs_eos)
        return {'outputs': enc_outputs, # (batch, seq_len, output_size)
                'lengths': lengths,
                'hiddens': hiddens}     # (num_layers * num_directions, batch, hidden_size)

    def decode(self, inputs, encoded):
        (decoded, lengths), hiddens = self.decoder(inputs, encoded['hiddens']) # (batch, max(dec_seq_lens), hidden_size)
        dec_outputs = self._flatten_and_unpad(decoded, lengths) # (n_tokens, hidden_size)
        return {'outputs': dec_outputs,
                'hiddens': hiddens}

    # used in beam_search()
    def _update_beam(self, input_tokens, next_tokens, encoded, decoded, parent_hypos, beam_searcher):
        input_tokens = next_tokens.unsqueeze(1)
        encoded['hiddens'] = self.decoder._reorder_hiddens(decoded['hiddens'], parent_hypos)
        encoded['outputs'] = encoded['outputs'][:beam_searcher.width]
        encoded['lengths'] = encoded['lengths'][:beam_searcher.width]
        return input_tokens, encoded

    # used in greedy_predict()
    def _get_last_hidden(self, decoded, batchsize, i):
        return decoded['outputs']

    def _update(self, input_tokens, output_tokens, encoded, decoded):
        input_tokens = output_tokens.view(-1, 1)
        encoded['hiddens'] = decoded['hiddens']
        return input_tokens, encoded


from .attentions import DotAttn, BiLinearAttn
class AttnSeq2Seq(Seq2Seq):
    # A fairly standard encoder-decoder architecture with the global attention mechanism in Luong et al. (2015).
    def __init__(self, embed_size, hidden_size, src_vocab_size, tgt_vocab_size,
                 src_EOS, tgt_BOS, tgt_EOS, num_layers=2, bidirectional='add', dropout=0.1, rnn='LSTM',
                 attention='bilinear', fuse_query='linear', init_w=0.1, beam_width=1, input_feeding=False):
        super().__init__(embed_size, hidden_size, src_vocab_size, tgt_vocab_size,
                         src_EOS, tgt_BOS, tgt_EOS,
                         num_layers=num_layers, bidirectional=bidirectional,
                         dropout=dropout, rnn=rnn, beam_width=beam_width, input_feeding=input_feeding)

        self.generator = MLP(dims=[self.hidden_size, tgt_vocab_size], dropout=dropout)
        assert attention in ['dot', 'bilinear']
        if attention == 'dot':
            self.attention = DotAttn(dim_q=hidden_size, dim_k=hidden_size, dim_v=hidden_size, fuse_query=fuse_query)
        elif attention == 'bilinear':
            self.attention = BiLinearAttn(dim_q=hidden_size, dim_k=hidden_size, dim_v=hidden_size, fuse_query=fuse_query)

        self.initialize(init_w)

    def decode(self, inputs, encoded):
        (decoded, dec_seq_lens), hiddens = self.decoder(inputs, encoded['hiddens']) # (batch, max(dec_seq_lens), hidden_size)
        # attention
        attn_vecs  = self.attention(queries=decoded, keys=encoded['outputs'], values=encoded['outputs'],
                                    query_lens=dec_seq_lens, key_lens=encoded['lengths'])  # (batch, max(dec_seq_lens), hidden_size)
        attn_vecs = self._flatten_and_unpad(attn_vecs, dec_seq_lens) # (n_tokens, hidden_size*2)
        return {'outputs': attn_vecs,
                'hiddens': hiddens}

    def decode_input_feeding(self, inputs, encoded):
        padded_seqs, dec_seq_lens, perm_idx = self.decoder.pad_and_sort_inputs(inputs)

        # compute outputs
        batch_sizes = self._lengths_to_batchsize(dec_seq_lens) #(max_len, )
        max_batch = len(inputs)
        decoded_list = []
        # sort encoded
        hiddens = self.decoder._reorder_hiddens(encoded['hiddens'], perm_idx)
        enc_outputs = encoded['outputs'][perm_idx]
        enc_lengths = encoded['lengths'][perm_idx]
        feed_vec = encoded['outputs'].new_zeros((max_batch, self.hidden_size)) if self.input_feeding else None
        for i, batch_size in enumerate(batch_sizes):
            batch_size = batch_size.item()
            hiddens = self.decoder._reorder_hiddens(hiddens, torch.arange(batch_size, dtype=torch.long))
            if self.input_feeding:
                feed_vec = feed_vec[:batch_size]
            decoded, hiddens = self.decoder.forward_step(padded_seqs[:batch_size, i], hiddens, feed_vec=feed_vec)
            # (batch, 1, num_directions * hidden_size), (num_layers * num_directions, batch, hidden_size)
            attn_vecs = self.attention(queries=decoded, keys=enc_outputs[:batch_size], values=enc_outputs[:batch_size],
                                       query_lens=enc_lengths.new_ones(batch_size), key_lens=enc_lengths[:batch_size])
            if self.input_feeding:
                feed_vec = attn_vecs.squeeze(1)
            if attn_vecs.size(0) < max_batch:
                # pad again
                attn_vecs = torch.cat((attn_vecs, attn_vecs.new_zeros((max_batch - attn_vecs.size(0), 1, attn_vecs.size(2)))), dim=0)
            decoded_list.append(attn_vecs)

        # sort into the original order
        _, unperm_idx = perm_idx.sort()
        decoded = torch.cat(decoded_list, dim=1)[unperm_idx] # (batch, max(dec_seq_lens), hidden_size)
        dec_seq_lens = dec_seq_lens[unperm_idx]
        decoded = self._flatten_and_unpad(decoded, dec_seq_lens)
        return decoded


    def _lengths_to_batchsize(self, seq_lens):
        n_tokens = torch.zeros(max(seq_lens).item(), dtype=torch.long)
        for l in seq_lens:
            n_tokens[:l] += 1
        return n_tokens
