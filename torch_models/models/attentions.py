import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DotAttn(nn.Module):
    def __init__(self, scaled=False, subsequent_mask=False):
        super().__init__()
        self.scaled = scaled
        self.subsequent_mask = subsequent_mask

    def forward(self, queries, keys, values, query_lens, key_lens):
        '''
        Parameters
        -----------
        queries : torch.tensor (batch, n_queries, dim1)
        keys : torch.tensor (batch, n_keys , dim1)
        values : torch.tensor (batch, n_keys, dim_value)
        query_lens : torch.LongTensor
            Contains the number of queries of each batch.
        key_lens : torch.LongTensor
            Contains the number of keys of each batch.
        '''
        # checks inputs
        self._check_len(queries, query_lens)
        self._check_len(keys, key_lens)
        # computes attention
        scores = torch.bmm(queries, keys.permute(0, 2, 1)) # (batch * n_queries * dim) @ (batch * dim * n_keys)
        if self.scaled:
            dim = queries.size(2)
            scores = scores/math.sqrt(dim)
        weights = self._masked_softmax(scores, query_lens, key_lens) # (batch, max(n_queries), max(n_keys))
        return torch.bmm(weights, values), weights # (batch, max(n_queries), dim_value)

    def _masked_softmax(self, scores, query_lens, key_lens):
        # masking
        masks = np.ones(scores.shape)
        for mask, q_len, k_len in zip(masks, query_lens, key_lens):
            mask[q_len:] = 0
            mask[:, k_len:] = 0
        if self.subsequent_mask: # used in self attention in decoder
            masks = np.tril(masks)
        masks = torch.from_numpy(masks).to(scores.device)
        scores.masked_fill_(masks==0, float('-inf'))

        weights = F.softmax(scores, dim=2)
        weights = torch.where(torch.isnan(weights), weights.new_tensor(0.), weights) # to avoid nan
        return weights

    def _check_len(self, tensor, lens):
        assert tensor.size(0) == lens.size(0)

# class BiLinearAttn(nn.Module):
#     def __init__(self, dim1, dim2):
#         self.linear = nn.Linear(dim1, dim2)
#
#     def __call__(self, keys, queries):
#         values = torch.where((keys == float('-inf')), torch.tensor(0.), keys)
#         queries = self.linear(queries)
#         scores = torch.bmm(queries, keys.permute(0, 2, 1)) # (batch * 1 * dim) @ (batch * dim * n_keys)
#         scores = torch.where(torch.isnan(scores), torch.tensor(float('-inf')), scores) # to avoid nan
#         weights = F.softmax(scores, dim=2)
#         return torch.bmm(weights, values)
