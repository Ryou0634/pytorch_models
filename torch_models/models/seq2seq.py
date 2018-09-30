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
        (outputs, lengths), hiddens = self.encoder.forward(inputs)
        if self.encoder.num_directions == 2:
            s = hiddens.shape # (num_layers * num_directions, batch, hidden_size)
            hiddens = hiddens.view(-1, 2, s[1], s[2]) # (num_layers, 2, batch, hidden_size)
            hiddens = torch.cat([hiddens[:, i] for i in range(2)], dim=2) # (num_layers, batch, hidden_size*2)
        return (outputs, lengths), hiddens

    def fit(self, inputs, targets, optimizer):
        self.train()
        # encoding
        _, enc_hiddens = self.encode(inputs) # (num_layers, batch, dec_hidden_size)
        # decoding
        BOS_targets = self._append_BOS(targets)
        (decoded, lengths), _ = self.decoder.forward(BOS_targets, enc_hiddens) # (batch, max(dec_seq_lens), dec_hidden_size)
        decoded = self._flatten_and_unpad(decoded, lengths) # (n_tokens, dec_hidden_size)
        # predicting
        targets_EOS = self._append_EOS_flatten(targets)
        loss = self.out_mlp.fit(decoded, targets_EOS, optimizer)
        return loss

    def generate(self, inputs, threshold=100):
        self.eval()
        with torch.no_grad():
            # encoding
            _, enc_hiddens = self.encode(inputs) # (num_layers * num_directions, batch, hidden_size)

            generated = []
            n_batch = enc_hiddens.shape[1]
            # decoding
            for i in range(n_batch):
                tgt_seq = []
                current_token = self.tgt_BOS
                hidden = enc_hiddens[:, i].unsqueeze(1) # computing batch by batch
                for _ in range(threshold):
                    (decoded, _), hidden = self.decoder.forward(torch.LongTensor([[current_token]]), hidden)
                    # predicting token
                    out = self.out_mlp.predict(decoded.squeeze(1)).item()
                    if out == self.tgt_EOS: break
                    tgt_seq.append(out)
                    current_token = out
                generated.append(tgt_seq)
        return generated

    def _append_EOS(self, inputs):
        inputs_EOS = [torch.cat((inp, torch.tensor([self.src_EOS]))) for inp in inputs]
        return inputs_EOS

    def _append_BOS(self, targets):
        BOS_targets = [torch.cat((torch.tensor([self.tgt_BOS]), target)) for target in targets]
        return BOS_targets

    def _append_EOS_flatten(self, targets):
        EOS_targets = [torch.cat((target, torch.tensor([self.tgt_EOS]))) for target in targets]
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

    def fit(self, inputs, targets, optimizer):
        self.train()
        # encoding
        (enc_outputs, enc_seq_lens), enc_hiddens = self.encode(inputs) # (batch, max(enc_seq_lens), dec_hidden_size), (num_layers, batch, dec_hidden_size)
        # decoding
        BOS_targets = self._append_BOS(targets)
        (decoded, dec_seq_lens), _ = self.decoder.forward(BOS_targets, enc_hiddens) # (batch, max(dec_seq_lens), dec_hidden_size)
        # attention
        weights = self.attention.forward(keys=enc_outputs, queries=decoded,
                                         keys_len=enc_seq_lens, queries_len=dec_seq_lens) # (batch, max(dec_seq_lens), max(enc_seq_lens))
        attn_vecs = torch.bmm(weights, enc_outputs) # (batch, max(dec_seq_lens), dec_hidden_size)
        # decoded + attention
        decoded_attn = torch.cat((decoded, attn_vecs), dim=2)
        decoded_attn = self._flatten_and_unpad(decoded_attn, dec_seq_lens) # (n_tokens, dec_hidden_size*2)
        # predicting
        targets_EOS = self._append_EOS_flatten(targets)
        loss = self.out_mlp.fit(decoded_attn, targets_EOS, optimizer)
        return loss

    def generate(self, inputs, threshold=100, attention=False):
        self.eval()
        with torch.no_grad():
            # encoding
            (enc_outputs, enc_seq_lens), enc_hiddens = self.encode(inputs) # (batch, max(enc_seq_lens), dec_hidden_size), (num_layers, batch, dec_hidden_size)

            generated = []
            n_batch = enc_hiddens.shape[1]
            if attention:
                attn_ws = [[] for _ in range(n_batch)]
            for i in range(n_batch):
                tgt_seq = []
                current_token = self.tgt_BOS
                enc_output = enc_outputs[i].unsqueeze(0) # (1, max(enc_seq_lens), dec_hidden_size)
                hidden = enc_hiddens[:, i].unsqueeze(1) # (num_layers, 1, dec_hidden_size)
                # decoding
                for _ in range(threshold):
                    (decoded, _), hidden = self.decoder.forward(torch.LongTensor([[current_token]]), hidden) # (1, 1, dec_hidden_size)
                    weights = self.attention.forward(enc_output, decoded, [enc_seq_lens[i]]) # (1, 1, max(enc_seq_lens)))
                    if attention:
                        attn_ws[i].append(weights)
                    # calculating attention
                    attn_vec = torch.bmm(weights, enc_output) # (1, 1, dec_hidden_size)
                    decoded_attn = torch.cat((decoded, attn_vec), dim=2)
                    # predicting token
                    out = self.out_mlp.predict(decoded_attn.squeeze(1)).item()
                    if out == self.tgt_EOS: break
                    tgt_seq.append(out)
                    current_token = out
                generated.append(tgt_seq)
            if attention:
                return generated, attn_ws
            else:
                return generated
