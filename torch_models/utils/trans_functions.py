import copy
import torch
import numpy as np


# trans_func for pytorch
# Used in my_utils.DataLoader.

class TransFunc():
    def __init__(self, device=torch.device('cpu')):
        self.device = device

    def __call__(self, batch):
        return

class torch_stack(TransFunc):
    def __call__(self, batch):
        inputs, targets = zip(*batch)
        inputs = torch.from_numpy(np.stack(inputs)).to(self.device)
        targets = torch.LongTensor(targets).to(self.device)
        return inputs, targets

class torch_seq(TransFunc):
    def __call__(self, batch):
        seqs = [torch.LongTensor(seq).to(self.device) for seq in batch]
        return seqs, None


class seq2label(TransFunc):
    def __call__(self, batch):
        inputs, targets = zip(*batch)
        seqs = [torch.LongTensor(seq).to(self.device) for seq in inputs]
        targets = torch.LongTensor(targets).to(self.device)
        return seqs, targets

class twoseq2label(TransFunc):
    def __call__(self, batch):
        inputs, targets = zip(*batch)
        seq1s, seq2s = zip(*inputs)
        seq1s = [torch.LongTensor(seq).to(self.device) for seq in seq1s]
        seq2s = [torch.LongTensor(seq).to(self.device) for seq in seq2s]
        targets = torch.LongTensor(targets).to(self.device)
        return (seq1s, seq2s), targets

class seq2seq(TransFunc):
    def __call__(self, batch):
        inputs, targets = zip(*batch)
        src_seq = [torch.LongTensor(seq).to(self.device) for seq in inputs]
        tgt_seq = [torch.LongTensor(seq).to(self.device) for seq in targets]
        return src_seq, tgt_seq

class twoseq2seq(TransFunc):
    def __call__(self, batch):
        inputs, targets = zip(*batch)
        seq1s, seq2s = zip(*inputs)
        seq1s = [torch.LongTensor(seq).to(self.device) for seq in seq1s]
        seq2s = [torch.LongTensor(seq).to(self.device) for seq in seq2s]
        targets = [torch.LongTensor(seq).to(self.device) for seq in targets]
        return (seq1s, seq2s), targets

class auto_encode(TransFunc):
    def __call__(self, batch):
        src_seq = [torch.LongTensor(seq).to(self.device) for seq in batch]
        tgt_seq = [torch.LongTensor(seq).to(self.device) for seq in batch]
        return src_seq, tgt_seq

class noising(TransFunc):
    def __init__(self, device=torch.device('cpu'), swap_rate=0.5, drop_rate=0):
        super().__init__(device)
        self.swap_rate = swap_rate
        self.drop_rate = drop_rate

    def __call__(self, batch):
        src_seq = [torch.LongTensor(self.contiguous_swap(self.drop(seq))).to(self.device)
                   for seq in batch]
        tgt_seq = [torch.LongTensor(seq).to(self.device) for seq in batch]
        return src_seq, tgt_seq

    def drop(self, seq):
        ''' Drop words randomly with a probability of self.drop_rate'''
        dropped = [s for s in seq if np.random.binomial(1, p=1-self.drop_rate)]
        return dropped

    def contiguous_swap(self, seq):
        '''
        For a sequence of N elements, the function
        makes N*swap_rate random swaps between contiguous tokens.
        '''
        seq = copy.copy(seq)
        seq_len = len(seq)
        for _ in range(int(seq_len*self.swap_rate)):
            i = np.random.randint(seq_len-1)
            seq[i], seq[i+1] = seq[i+1], seq[i]
        return seq
