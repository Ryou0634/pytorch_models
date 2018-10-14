import torch

def get_device(show_log=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if show_log:
        print('===== Device =====')
        print(device)
    return device

def initialize_embed():
    # TODO
    return

def get_closest_word_vecotors(word, embedding, vocab_dict, topk=10):
    '''
    Get a list of the top-k closest word vecotors.

    Parameters
    ----------
    word : str
        target word.
    embedding : torch.tensor
        2d-tensor of word embeddings. (vocab_size, embedding_dim)
    vocab_dict : my_utils.Dictionary
        Dictionary to look up a word by index, and vice versa.
    '''
    idx = vocab_dict(word)
    target = embedding[idx]
    norm = torch.norm(embedding, dim=1)*torch.norm(target)
    sim = (torch.matmul(embedding, target.view(-1, 1)).squeeze())/norm
    sims, idxs = torch.topk(sim, k=topk)
    for sim, idx in zip(sims, idxs):
        print('{:.4}\t{}'.format(float(sim), vocab_dict(int(idx))))
    return
