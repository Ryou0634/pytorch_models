from my_utils import seq_10, DataLoader
from torch_models.utils import seq2label, get_device

train = seq_10(5000)
test = seq_10(500)
device = get_device()
trans_func = seq2label(device)
train_loader = DataLoader(train, batch_size=16, trans_func=trans_func)
test_loader = DataLoader(test, batch_size=64, trans_func=trans_func)


from my_utils import Trainer, EvaluatorC
from torch.optim import Adam

from torch_models.models import BoV, RNNLastHidden, RNNMaxPool, SingleClassifier
def test_BoV():
    encoder = BoV(embed_size=50, vocab_size=10).to(device)
    model = SingleClassifier(encoder=encoder, output_size=2, hidden_size=None,
                             activation='Tanh', dropout=0, freeze_encoder=False).to(device)

    optimizer = Adam(model.parameters())

    evaluator = EvaluatorC(model, test_loader)
    trainer = Trainer(model, train_loader)
    trainer.train_epoch(optimizer, max_epoch=1,
                  evaluator=evaluator, score_monitor=None)
    assert evaluator.evaluate() > 0.8

import pytest
@pytest.mark.parametrize(
    'rnn, enc, bidirectional, num_layers', [
        ('RNN', RNNMaxPool, 'cat', 2),
        ('GRU', RNNMaxPool, 'add', 1),
        ('LSTM', RNNLastHidden, None, 1),
    ]
)
def test_RNN(rnn, enc, bidirectional, num_layers):
    encoder = enc(embed_size=10, hidden_size=15, vocab_size=10, bidirectional=bidirectional,
                  num_layers=num_layers, rnn=rnn).to(device)
    model = SingleClassifier(encoder=encoder, output_size=2, hidden_size=None,
                             activation='Tanh', dropout=0, freeze_encoder=False).to(device)
    optimizer = Adam(model.parameters())

    evaluator = EvaluatorC(model, test_loader)
    trainer = Trainer(model, train_loader)
    trainer.train_epoch(optimizer, max_epoch=1,
                  evaluator=evaluator, score_monitor=None)
    assert evaluator.evaluate() > 0.85
