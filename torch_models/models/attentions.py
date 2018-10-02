import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DotAttn():
    def __init__(self, scaled=False, device='cpu'):
        self.scaled = scaled
        self.device = torch.device(device)

    def forward(self, keys, queries, keys_len=None, queries_len=None):
        '''
        Parameters
        -----------
        keys : torch.tensor (batch * n_keys * dim) (this is also regarded as values)
        queries : torch.tensor (batch * n_queries * dim)
        keys_len : List[int]
            Contains the number of keys of each batch.
        queries_len : List[int]
            Contains the number of queries of each batch.
        '''
        if keys_len is None:
            keys_len = [keys.shape[1] for _ in range(keys.shape[0])]
        if queries_len is None:
            queries_len = [queries.shape[1] for _ in range(queries.shape[0])]
        scores = torch.bmm(queries, keys.permute(0, 2, 1)) # (batch * n_queries * dim) @ (batch * dim * n_keys)
        if self.scaled:
            dim = queries.shape[2]
            scores = scores/math.sqrt(dim)
        weights = self._masked_softmax(scores, keys_len, queries_len)
        return weights # (batch, max(n_queries), max(n_keys))

    def _masked_softmax(self, scores, keys_len, queries_len):
        for score, q_len, k_len in zip(scores, queries_len, keys_len):
            score[q_len:] = torch.tensor(float('-inf'))
            score[:, k_len:] = torch.tensor(float('-inf'))
        weights = F.softmax(scores, dim=2)
        weights = torch.where(torch.isnan(weights), torch.tensor(0., device=self.device), weights) # to avoid nan
        return weights

class BiLinearAttn(nn.Module):
    def __init__(self, dim1, dim2, device='cpu'):
        self.linear = nn.Linear(dim1, dim2)
        self.device = torch.device(device)
        self.to(self.device)

    def forward(self, keys, queries):
        values = torch.where((keys == float('-inf')), torch.tensor(0.), keys)
        queries = self.linear(queries)
        scores = torch.bmm(queries, keys.permute(0, 2, 1)) # (batch * 1 * dim) @ (batch * dim * n_keys)
        scores = torch.where(torch.isnan(scores), torch.tensor(float('-inf')), scores) # to avoid nan
        weights = F.softmax(scores, dim=2)
        return torch.bmm(weights, values)
