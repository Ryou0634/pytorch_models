import torch
import numpy as np
from torch_models.models import DotAttn


batchsize = 4
max_n_q = 4
max_n_k = 3
dim = 4
queries = torch.ones((batchsize, max_n_q, dim))
keys = torch.ones((batchsize, max_n_k, dim))
values = torch.ones((batchsize, max_n_k, dim))
query_lens = torch.LongTensor([4, 1, 2, 3])
key_lens = torch.LongTensor([3, 2, 1, 3])

def test_forward():
    attn = DotAttn(dim_q=4, dim_k=4, dim_v=4, scaled=False, subsequent_mask=False)
    outputs = attn(queries=queries, keys=keys, values=values, query_lens=query_lens, key_lens=key_lens)
    expected_w = np.array([
                           [[1/3, 1/3, 1/3],
                            [1/3, 1/3, 1/3],
                            [1/3, 1/3, 1/3],
                            [1/3, 1/3, 1/3]],
                           [[1/2, 1/2, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]],
                           [[1, 0, 0],
                            [1, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]],
                           [[1/3, 1/3, 1/3],
                            [1/3, 1/3, 1/3],
                            [1/3, 1/3, 1/3],
                            [0, 0, 0]]
                           ])
    assert np.allclose(expected_w, attn.weights.detach().numpy())


def test_subsequent_mask():
    attn = DotAttn(dim_q=4, dim_k=4, dim_v=4, scaled=False, subsequent_mask=True)
    attn.train()
    outputs = attn(queries=queries, keys=keys, values=values, query_lens=query_lens, key_lens=key_lens)
    expected_w = np.array([
                           [[1, 0, 0],
                            [1/2, 1/2, 0],
                            [1/3, 1/3, 1/3],
                            [1/3, 1/3, 1/3]],
                           [[1, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]],
                           [[1, 0, 0],
                            [1, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]],
                           [[1, 0, 0],
                            [1/2, 1/2, 0],
                            [1/3, 1/3, 1/3],
                            [0, 0, 0]]
                           ])
    assert np.allclose(expected_w, attn.weights.detach().numpy())
