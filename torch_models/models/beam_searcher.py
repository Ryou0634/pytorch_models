import torch
import numpy as np

class BeamSearcher():
    def __init__(self, width, eos):
        self.width = width
        self.eos = eos
        self.hypos  = [{'seq': [], 'score': 0} for _ in range(width)]
        self.parent_hypos = [i for i in range(width)]
        self.end_hypos = []

    def step(self, step, log_p):
        scores, top_seqs = log_p.topk(self.width, dim=1)
        for score, parent_hypo in zip(scores, self.parent_hypos):
            score += self.hypos[parent_hypo]['score']
        # get top k scores
        if step == 0:
            idxs = [i for i in range(self.width)]
        else:
            flattened_axis = np.argsort(-scores, axis=None)[:self.width]
            self.parent_hypos = flattened_axis/scores.size(1)
            idxs = flattened_axis%scores.size(1)
        # select new hypotheses
        new_hypos = []
        new_parent_hypos = []
        old_parent_hypos = []
        next_tokens = []
        for parent_hypo, idx in zip(self.parent_hypos, idxs):
            next_token = int(top_seqs[parent_hypo][idx])
            if next_token == self.eos:
                self.end_hypos.append({'score': scores[parent_hypo][idx],
                                       'seq': self.hypos[parent_hypo]['seq']})
                self.width -= 1
            else:
                next_tokens.append(next_token)
                old_parent_hypos.append(parent_hypo)
                new_parent_hypos.append(len(new_hypos))
                new_hypos.append({'score': scores[parent_hypo][idx],
                                  'seq': self.hypos[parent_hypo]['seq']+[next_token]})
        next_tokens = torch.tensor(next_tokens)
        self.hypos = new_hypos
        self.parent_hypos = new_parent_hypos
        return next_tokens, old_parent_hypos
