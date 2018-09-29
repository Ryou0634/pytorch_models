import torch
import torch.nn as nn
import torch.nn.functional as F

class DotAttn():
    def __init__(self, scaled=False):
        self.scaled = scaled

    def forward(self, keys, queries):
        '''
        Parameters
        -----------
        keys : torch.tensor (batch * n_keys * dim)
        queries : torch.tensor (batch * 1 * dim)
        '''
        values = torch.where((keys == float('-inf')), torch.tensor(0.), keys)
        scores = torch.bmm(queries, keys.permute(0, 2, 1)) # (batch * 1 * dim) @ (batch * dim * n_keys)
        if self.scaled:
            dim = queries.shape[2]
            score = score/math.sqrt(dim)
        scores = torch.where(torch.isnan(scores), torch.tensor(float('-inf')), scores) # to avoid nan
        weights = F.softmax(scores, dim=2)
        return torch.bmm(weights, values)

class BiLinearAttn():
    def __init__(self, dim1, dim2):
        self.linear = nn.Linear(dim1, dim2)

    def forward(self, keys, queries):
        values = torch.where((keys == float('-inf')), torch.tensor(0.), keys)
        queries = self.linear(queries)
        scores = torch.bmm(queries, keys.permute(0, 2, 1)) # (batch * 1 * dim) @ (batch * dim * n_keys)
        scores = torch.where(torch.isnan(scores), torch.tensor(float('-inf')), scores) # to avoid nan
        weights = F.softmax(scores, dim=2)
        return torch.bmm(weights, values)
