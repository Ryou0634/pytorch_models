import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBase(nn.Module):
    def __init__(self, dim_q=None, dim_k=None, dim_v=None,
                 subsequent_mask=False, fuse_query=None):
        '''
        Parameters
        -----------
        dim_q : int
            This must be specified in BiLinearAttn or when fuse_query = 'linear'.
        dim_k : int
            This must be specified in BiLinearAttn.
        dim_v : int
            This must be specified when fuse_query = 'linear'.
        subsequent_mask : bool
            If True, apply subsequent mask when computing softmax.
        fuse_query : None, 'linear', 'add'
            Specify how to fuse attention vectors and query vectors.
        '''
        super().__init__()
        self.subsequent_mask = subsequent_mask

        assert fuse_query in [None, 'linear', 'add']
        self.fuse_query = fuse_query
        if self.fuse_query == 'linear':
            self.fuse_W = nn.Linear(dim_v+dim_q, dim_v)
        self.weights = None

    def forward(self, queries, keys, values, query_lens, key_lens):
        '''
        Parameters
        -----------
        queries : torch.tensor (batch, n_queries, dim_q)
        keys : torch.tensor (batch, n_keys , dim_k)
        values : torch.tensor (batch, n_keys, dim_v)
        query_lens : torch.LongTensor
            Contains the number of queries of each batch.
        key_lens : torch.LongTensor
            Contains the number of keys of each batch.
        '''
        # checks inputs
        self._check_len(queries, query_lens)
        self._check_len(keys, key_lens)
        # computes attention
        scores = self.compute_score(queries, keys)
        weights = self._masked_softmax(scores, query_lens, key_lens) # (batch, max(query_lens), max(n_keys))
        self.weights = weights # for debugging or analyzing
        attn_vecs = torch.bmm(weights, values) # (batch, max(key_lens), dim_v)
        if self.fuse_query:
            return self._fuse_attn_query(attn_vecs, queries)
        else:
            return attn_vecs

    def _fuse_attn_query(self, attn_vecs, queries):
        if self.fuse_query == 'linear':
            fused = torch.tanh(self.fuse_W(torch.cat((queries, attn_vecs), dim=2)))
        elif self.fuse_query == 'add':
            fused = queries + attn_vecs
        return fused

    def compute_score(self, queries, keys):
        raise NotImplementedError()

    def _check_len(self, tensor, lens):
        assert tensor.size(0) == lens.size(0)

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

class DotAttn(AttentionBase):
    def __init__(self, dim_q=None, dim_k=None, dim_v=None, subsequent_mask=None, fuse_query=None, scaled=False):
        super().__init__(dim_q=dim_q, dim_k=dim_k, dim_v=dim_v,
                         subsequent_mask=subsequent_mask, fuse_query=fuse_query)
        self.scaled = scaled

    def compute_score(self, queries, keys):
        scores = torch.bmm(queries, keys.permute(0, 2, 1)) # (batch * n_queries * dim_q) @ (batch * dim_k * n_keys)
        if self.scaled:
            dim = queries.size(2)
            scores = scores/math.sqrt(dim)
        return scores

class BiLinearAttn(AttentionBase):
    def __init__(self, dim_q, dim_k, dim_v=None, subsequent_mask=None, fuse_query='linear'):
        super().__init__(dim_q=dim_q, dim_k=dim_k, dim_v=dim_v,
                         subsequent_mask=subsequent_mask, fuse_query=fuse_query)
        self.linear = nn.Linear(dim_q, dim_k, bias=False)


    def compute_score(self, queries, keys):
        queries = self.linear(queries) # (batch, n_queries, dim_q) @ (dim_q, dim_k)
        scores = torch.bmm(queries, keys.permute(0, 2, 1)) # (batch, n_queries, dim_k) @ (batch, dim_k, n_keys)
        return scores
