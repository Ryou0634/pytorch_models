import torch
import numpy as np

def get_device(show_log=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if show_log:
        print('===== Device =====')
        print(device)
    return device

# trans_func for pytorch
# Used in my_utils.DataLoader.
def torch_seq(inputs, targets):
    device = get_device(show_log=False)
    seqs = [torch.LongTensor(seq).to(device) for seq in inputs]
    targets = torch.LongTensor(targets).to(device)
    return seqs, targets

def torch_two_seqs(inputs, targets):
    device = get_device(show_log=False)
    seq1s, seq2s = zip(*inputs)
    seq1s = [torch.LongTensor(seq).to(device) for seq in seq1s]
    seq2s = [torch.LongTensor(seq).to(device) for seq in seq2s]
    targets = torch.LongTensor(targets).to(device)
    return (seq1s, seq2s), targets

def seqs_to_seq(inputs, targets):
    device = get_device(show_log=False)
    seq1s, seq2s = zip(*inputs)
    seq1s = [torch.LongTensor(seq).to(device) for seq in seq1s]
    seq2s = [torch.LongTensor(seq).to(device) for seq in seq2s]
    targets = [torch.LongTensor(seq).to(device) for seq in targets]
    return (seq1s, seq2s), targets

def torch_stack(inputs, targets):
    device = get_device(show_log=False)
    inputs = torch.from_numpy(np.stack(inputs)).to(device)
    targets = torch.LongTensor(targets).to(device)
    return inputs, targets
