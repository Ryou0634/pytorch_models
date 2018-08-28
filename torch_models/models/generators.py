import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .seq_encoders import LSTMEncoderBase

class Generator(LSTMEncoderBase):
    def __init__(self, vocab_dict, embed_size):
        self.vocab_dict = vocab_dict
        super().__init__(embed_size=embed_size, vocab_size=len(vocab_dict),
                         bidirectional=False, num_layers=1, gru=True)
        self.rnn2out = nn.Linear(self.output_size, self.vocab_size)

    def forward(self, inputs, hidden=None):
        embed_seqs = self._get_embeds(inputs)
        packed_seqs, original_idx = self._pack_embeds(embed_seqs)

        outputs, _ = self.rnn(packed_seqs, hidden)
        tensors, lengths = pad_packed_sequence(outputs, batch_first=True)
        tensors = self.rnn2out(tensors)
        outputs = []
        for i in range(len(lengths)):
            outputs.append(tensors[i][:lengths[i]])
        outputs = self._reorder_batch(outputs, original_idx)
        return outputs

    def _reorder_batch(self, tensors, original_idx):
        _, idxs = torch.sort(torch.tensor(original_idx))
        tensors = [tensors[idx] for idx in idxs]
        return tensors


class ConditionedGenerator(nn.Module):
    # generator is conditioned by inputting the output of encoder to the initial hidden state
    def __init__(self, encoder, generator):
        super().__init__()
        self.encoder = encoder
        self.generator = generator
        self.vocab_dict = generator.vocab_dict

        self.criterion = nn.CrossEntropyLoss()
        self.gumbel = torch.distributions.gumbel.Gumbel(0, 1)

    def fit(self, inputs, targets, optimizer):
        self.zero_grad()

        conditions, words = inputs
        init_hidden = self.encoder.forward(conditions).unsqueeze(0)
        outputs = self.generator.forward(words, init_hidden)
        outputs = torch.cat(outputs, dim=0)
        targets = torch.cat(targets)
        loss = self.criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        return loss.item()

    def forward(self, current_word, hidden):
        embed = self.generator.embedding.weight[current_word].view(1, 1, -1)
        output, hidden = self.generator.rnn(embed, hidden)
        output = self.generator.rnn2out(output)[0]
        output = F.softmax(output, dim=1)
        return output

    def generate(self, condition, beam_width=1, max_length=10):
        seq = []
        hidden = self.encoder.forward(condition).view(1, 1, -1)
        current_word = self.vocab_dict('<SOS>')

        for _ in range(max_length):
            p = self.forward(current_word, hidden)[0]
            current_word = np.random.choice(len(p), p=p.detach().numpy())
            if current_word == self.vocab_dict('<EOS>'):
                break
            seq.append(current_word)
        return ' '.join([self.vocab_dict(idx) for idx in seq])
