from torch_models.models import MLP, LSTMEncoder
import torch.nn as nn
import torch

class Seq2Seq(nn.Module):
    def __init__(self, embed_size, hidden_size, src_vocab_size, tgt_vocab_size,
                 src_EOS, tgt_BOS, tgt_EOS, num_layers=1, bidirectional=False):
        super().__init__()
        self.encoder = LSTMEncoder(embed_size, hidden_size, src_vocab_size,
                                   bidirectional=bidirectional, num_layers=num_layers)
        self.dec_hidden_size = hidden_size*(1+bidirectional)
        self.decoder = LSTMEncoder(embed_size, self.dec_hidden_size, tgt_vocab_size,
                                   bidirectional=False, num_layers=num_layers)
        self.out_mlp = MLP(dims=[self.dec_hidden_size, tgt_vocab_size])

        self.src_EOS = src_EOS
        self.tgt_BOS = tgt_BOS
        self.tgt_EOS = tgt_EOS

    def encode(self, inputs):
        inputs = self._append_EOS(inputs)
        (outputs, lengths), (hiddens, cells) = self.encoder(inputs)
        if self.encoder.num_directions == 2:
            hiddens = self._concat_bi_direction(hiddens)
            cells = self._concat_bi_direction(cells)
        return (outputs, lengths), (hiddens, cells)

    def _concat_bi_direction(self, hiddens):# (num_layers * num_directions, batch, hidden_size)
        s = hiddens.shape # (num_layers * num_directions, batch, hidden_size)
        hiddens = hiddens.view(-1, 2, s[1], s[2]) # (num_layers, 2, batch, hidden_size)
        hiddens = torch.cat([hiddens[:, i] for i in range(2)], dim=2) # (num_layers, batch, hidden_size*2)
        return hiddens

    def decode(self, inputs, enc_hiddens, enc_cells, enc_outputs, enc_seq_lens):
        (decoded, lengths), (hidden, cells) = self.decoder(inputs, (enc_hiddens, enc_cells)) # (batch, max(dec_seq_lens), dec_hidden_size)
        decoded = self._flatten_and_unpad(decoded, lengths) # (n_tokens, dec_hidden_size)
        return decoded, (hidden, cells)

    def fit(self, inputs, targets, optimizer):
        self.train()
        self.zero_grad()
        # encoding
        (enc_outputs, enc_seq_lens), (enc_hiddens, enc_cells) = self.encode(inputs) # (num_layers, batch, dec_hidden_size)
        # decoding
        BOS_targets = self._append_BOS(targets)
        decoded, _ = self.decode(BOS_targets, enc_hiddens, enc_cells, enc_outputs, enc_seq_lens)
        # predicting
        targets_EOS = self._append_EOS_flatten(targets)
        loss = self.out_mlp.fit(decoded, targets_EOS, optimizer)
        return loss

    def predict(self, inputs, max_len=100):
        self.eval()
        generated = []
        with torch.no_grad():
            # encoding
            (enc_outputs, enc_seq_lens), (hiddens, cells) = self.encode(inputs) # (num_layers * num_directions, batch, hidden_size)
            batchsize = hiddens.shape[1]
            input_tokens = inputs[0].new_tensor([self.tgt_BOS for _ in range(batchsize)]).view(-1, 1)
            end_flags = inputs[0].new_zeros(batchsize)
            for i in range(max_len):
                decoded, (hiddens, cells) = self.decode(input_tokens, hiddens, cells, enc_outputs, enc_seq_lens) # (n_tokens, dec_hidden_size)
                output_tokens = self.out_mlp.predict(decoded)
                generated.append(output_tokens)
                end_flags.masked_fill_(output_tokens.eq(self.tgt_EOS), 1) # set 1 in end_flags if EOS
                if end_flags.sum() == batchsize: break
                input_tokens = output_tokens.view(-1, 1)
        generated = torch.stack(generated, dim=1).tolist()
        return self._remove_EOS(generated)

    def _remove_EOS(self, generated):
        outputs = []
        for seq in generated:
            if self.tgt_EOS in seq:
                EOS_idx = seq.index(self.tgt_EOS)
                outputs.append(seq[:EOS_idx])
            else:
                outputs.append(seq)
        return outputs

    def _append_EOS(self, inputs):
        inputs_EOS = [torch.cat((inp, inp.new_tensor([self.src_EOS]))) for inp in inputs]
        return inputs_EOS

    def _append_BOS(self, targets):
        BOS_targets = [torch.cat((target.new_tensor([self.tgt_BOS]), target)) for target in targets]
        return BOS_targets

    def _append_EOS_flatten(self, targets):
        EOS_targets = [torch.cat((target, target.new_tensor([self.tgt_EOS]))) for target in targets]
        return torch.cat(EOS_targets)

    def _flatten_and_unpad(self, decoded, lengths):
        # (batch, max_length, embed_dim)
        unpadded = [batch[:l] for batch, l in zip(decoded, lengths)]
        flattened = torch.cat(unpadded, dim=0)
        return flattened

from .attentions import DotAttn

class AttnSeq2Seq(Seq2Seq):
    def __init__(self, embed_size, hidden_size, src_vocab_size, tgt_vocab_size,
                 src_EOS, tgt_BOS, tgt_EOS, num_layers=1, bidirectional=False):
        super().__init__(embed_size, hidden_size, src_vocab_size, tgt_vocab_size,
                         src_EOS, tgt_BOS, tgt_EOS, num_layers, bidirectional)
        self.out_mlp = MLP(dims=[self.dec_hidden_size*2, tgt_vocab_size])
        self.attention = DotAttn(scaled=True)
        self.get_attn = False
        self.attn_weights = []

    def decode(self, inputs, enc_hiddens, enc_cells, enc_outputs, enc_seq_lens):
        (decoded, dec_seq_lens), (hiddens, cells) = self.decoder(inputs, (enc_hiddens, enc_cells)) # (batch, max(dec_seq_lens), dec_hidden_size)
        # attention
        weights = self.attention(keys=enc_outputs, queries=decoded,
                                 keys_len=enc_seq_lens, queries_len=dec_seq_lens) # (batch, max(dec_seq_lens), max(enc_seq_lens))
        if self.get_attn:
            self.attn_weights.append(weights)
        attn_vecs = torch.bmm(weights, enc_outputs) # (batch, max(dec_seq_lens), dec_hidden_size)
        # decoded + attention
        decoded_attn = torch.cat((decoded, attn_vecs), dim=2)
        decoded_attn = self._flatten_and_unpad(decoded_attn, dec_seq_lens) # (n_tokens, dec_hidden_size*2)
        return decoded_attn, (hiddens, cells)
