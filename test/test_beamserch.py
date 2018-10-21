import torch
from torch_models import BeamSearcher


def test_bm():
    bm = BeamSearcher(width=2, eos=0)

    scores1 = torch.tensor([[0, 4, 3],
                        [0, 3, 5]])
    scores2 = torch.tensor([[2, 5, 7],
                            [2, 3, 3]])
    scores3 = torch.tensor([[5, 4, 3],
                            [7, 4, 3]])

    bm.step(0, scores1)
    assert bm.hypos[0] == {'score': torch.tensor(4), 'seq': [1]}
    assert bm.hypos[1] == {'score': torch.tensor(3), 'seq': [1]}
    bm.step(1, scores2)
    assert bm.hypos[0] == {'score': torch.tensor(11), 'seq': [1, 2]}
    assert bm.hypos[1] == {'score': torch.tensor(9), 'seq': [1, 1]}
    bm.step(2, scores3)
    assert len(bm.hypos) == 0
    assert bm.end_hypos[0] == {'score': torch.tensor(16), 'seq': [1, 2]}
    assert bm.end_hypos[1] == {'score': torch.tensor(16), 'seq': [1, 1]}
