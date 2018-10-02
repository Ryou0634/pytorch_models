import torch
import numpy as np

# trans_func for pytorch
# Used in my_utils.DataLoader.

class TransFunc():
    def __init__(self, device='cpu'):
        self.device = torch.device(device)

    def __call__(self, inputs, targets):
        return

class seq2label(TransFunc):
    def __call__(self, inputs, targets):
        seqs = [torch.LongTensor(seq).to(self.device) for seq in inputs]
        targets = torch.LongTensor(targets).to(self.device)
        return seqs, targets

class twoseq2label(TransFunc):
    def __call__(self, inputs, targets):
        seq1s, seq2s = zip(*inputs)
        seq1s = [torch.LongTensor(seq).to(self.device) for seq in seq1s]
        seq2s = [torch.LongTensor(seq).to(self.device) for seq in seq2s]
        targets = torch.LongTensor(targets).to(self.device)
        return (seq1s, seq2s), targets

class seq2seq(TransFunc):
    def __call__(self, inputs, targets):
        src_seq = [torch.LongTensor(seq).to(self.device) for seq in inputs]
        tgt_seq = [torch.LongTensor(seq).to(self.device) for seq in targets]
        return src_seq, tgt_seq

class twoseq2seq(TransFunc):
    def __call__(self, inputs, targets):
        seq1s, seq2s = zip(*inputs)
        seq1s = [torch.LongTensor(seq).to(self.device) for seq in seq1s]
        seq2s = [torch.LongTensor(seq).to(self.device) for seq in seq2s]
        targets = [torch.LongTensor(seq).to(self.device) for seq in targets]
        return (seq1s, seq2s), targets

class torch_stack(TransFunc):
    def __call__(self, inputs, targets):
        inputs = torch.from_numpy(np.stack(inputs)).to(self.device)
        targets = torch.LongTensor(targets).to(self.device)
        return inputs, targets
