from torch_models.models import MLP, RNNEncoder, BeamSearcher
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

    def _remove_EOS(self, generated):
        outputs = []
        for seq in generated:
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
        return flattened # (n_tokens, embed_dim)

class Seq2Seq(Seq2SeqBase):
    def __init__(self, embed_size, hidden_size, src_vocab_size, tgt_vocab_size,
                 src_EOS, tgt_BOS, tgt_EOS, num_layers=1, bidirectional=False, dropout=0, rnn='LSTM',
                 init_w=0.1, beam_width=1):
        if bidirectional:
            bidir_type = 'add'
        else:
            bidir_type = None
        super().__init__(src_EOS, tgt_BOS, tgt_EOS, beam_width)
        self.encoder = RNNEncoder(embed_size, hidden_size, src_vocab_size,
                                  bidirectional=bidir_type, num_layers=num_layers, dropout=dropout, rnn=rnn)
        self.hidden_size = hidden_size
        self.decoder = RNNEncoder(embed_size, self.hidden_size, tgt_vocab_size,
                                   bidirectional=None, num_layers=num_layers, dropout=dropout, rnn=rnn)
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


    def beam_search(self, input, max_len=100):
        """input allows only batchsize 1"""
        self.eval()
        beam_searcher = BeamSearcher(self.beam_width, self.tgt_EOS)

        with torch.no_grad():
            # encoding
            encoded = self.encode(input.repeat(self.beam_width, 1))
            input_tokens = input[0].new_tensor([[self.tgt_BOS] for _ in range(self.beam_width)]) # (beam_width, 1)
            for i in range(max_len):
                decoded  = self.decode(input_tokens, encoded) # (beam_width, hidden_size)
                log_p = F.log_softmax(self.generator.forward(decoded['outputs']), dim=1) # (beam_width, tgt_vocab_size)
                next_tokens, parent_hypos = beam_searcher.step(i, log_p)
                if beam_searcher.width == 0: break
                # preparing for the next iteration
                input_tokens = next_tokens.view(beam_searcher.width, 1)
                encoded['hiddens'] = self.decoder._reorder_hiddens(decoded['hiddens'], parent_hypos)
                encoded['outputs'] = encoded['outputs'][:beam_searcher.width]
                encoded['lengths'] = encoded['lengths'][:beam_searcher.width]
        return beam_searcher.hypos + beam_searcher.end_hypos

    def predict(self, inputs, max_len=100):
        predicted = []
        for inpt in inputs:
            top_seqs = self.beam_search(inpt, max_len)
            top_seq = max(top_seqs, key=lambda x: x['score'])['seq']
            predicted.append(top_seq)
        return predicted

    def greedy_predict(self, inputs, max_len=50):
        self.eval()
        generated = [] # contains selected idxs in each time step
        with torch.no_grad():
            # encoding
            encoded = self.encode(inputs) # (num_layers * num_directions, batch, hidden_size)
            batchsize = len(inputs)
            input_tokens = inputs[0].new_tensor([self.tgt_BOS for _ in range(batchsize)]).view(-1, 1) # (batchsize, 1)
            end_flags = inputs[0].new_zeros(batchsize)
            for i in range(max_len):
                decoded  = self.decode(input_tokens, encoded) # (batchsize, hidden_size)
                output_tokens = self.generator.predict(decoded['outputs']) # (batchsize,)
                generated.append(output_tokens)
                end_flags.masked_fill_(output_tokens.eq(self.tgt_EOS), 1) # set 1 in end_flags if EOS
                if end_flags.sum() == batchsize: break # end if all flags are 1
                input_tokens = output_tokens.view(-1, 1)
                encoded['hiddens'] = decoded['hiddens']
        generated = torch.stack(generated, dim=1)
        return self._remove_EOS(generated)


from .attentions import DotAttn
class AttnSeq2Seq(Seq2Seq):
    # A fairly standard encoder-decoder architecture with the global attention mechanism in Luong et al. (2015).
    def __init__(self, embed_size, hidden_size, src_vocab_size, tgt_vocab_size,
                 src_EOS, tgt_BOS, tgt_EOS, num_layers=1, bidirectional=False, dropout=0, rnn='LSTM',
                 attention='dot', attn_hidden='linear', init_w=0.1, beam_width=1):
        super().__init__(embed_size, hidden_size, src_vocab_size, tgt_vocab_size,
                         src_EOS, tgt_BOS, tgt_EOS,
                         num_layers=num_layers, bidirectional=bidirectional,
                         dropout=dropout, rnn=rnn, beam_width=beam_width)

        self.generator = MLP(dims=[self.hidden_size, tgt_vocab_size], dropout=dropout)
        if attention == 'dot':
            self.attention = DotAttn()
        self.attn_weights = None

        if attn_hidden == 'linear':
            self.attn_hidden = nn.Linear(self.hidden_size*2, self.hidden_size)
        elif attn_hidden == 'add':
            self.attn_hidden = None
        else:
            raise Exception("attn_hidden: ['linear', 'add']")

        self.initialize(init_w)

    def decode(self, inputs, encoded):
        (decoded, dec_seq_lens), hiddens = self.decoder(inputs, encoded['hiddens']) # (batch, max(dec_seq_lens), hidden_size)
        # attention
        attn_vecs, self.attn_weights = self.attention(queries=decoded, keys=encoded['outputs'], values=encoded['outputs'],
                                       query_lens=dec_seq_lens, key_lens=encoded['lengths'])  # (batch, max(dec_seq_lens), hidden_size)

        # decoded + attention
        if self.attn_hidden:
            decoded_attn = torch.tanh(self.attn_hidden(torch.cat((decoded, attn_vecs), dim=2)))
        else:
            decoded_attn = decoded + attn_vecs
        decoded_attn = self._flatten_and_unpad(decoded_attn, dec_seq_lens) # (n_tokens, hidden_size*2)
        return {'outputs': decoded_attn,
                'hiddens': hiddens}
