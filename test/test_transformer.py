from my_utils import Dictionary, DataLoader, Trainer, EvaluatorSeq
from torch_models.utils import seq2seq
from my_utils.toy_data import invert_seq

n_unique = 10

src_dict = Dictionary(['<EOS>'])
tgt_dict = Dictionary(['<BOS>', '<EOS>'])
for n in range(n_unique):
    src_dict.add_word(str(n))
    tgt_dict.add_word(str(n))

train = invert_seq(5000, n_unique=n_unique)
test = invert_seq(100, n_unique=n_unique)


def numericalize(dataset, src_dict, tgt_dict):
    numericalized = [([src_dict(s) for s in src], [tgt_dict(t) for t in tgt]) for src, tgt in dataset]
    return numericalized

trans_func = seq2seq()
train_loader = DataLoader(numericalize(train, src_dict, tgt_dict), batch_size=64, trans_func=trans_func)
test_loader = DataLoader(numericalize(test, src_dict, tgt_dict), batch_size=10, trans_func=trans_func)

from torch_models.models.transformer import Transformer
from torch.optim import Adam, SGD
embed_size = 24
n_head = 4

def test_tf():
    model = Transformer(size=embed_size, n_head=n_head, src_vocab_size=len(src_dict), tgt_vocab_size=len(tgt_dict),
                    src_EOS=src_dict('<EOS>'), tgt_BOS=tgt_dict('<BOS>'), tgt_EOS=tgt_dict('<EOS>'),
                    dropout=0, n_layers=1)

    optimizer = Adam(model.parameters())

    trainer = Trainer(model, train_loader)
    trainer.train_epoch(optimizer, max_epoch=5,
                        evaluator=None, score_monitor=None)
    evaluator = EvaluatorSeq(model, test_loader, measure='BLEU')
    assert evaluator.evaluate() > 0.85
