from my_utils import seq_10, DataLoader
from torch_models.utils import seq2label, get_device

train = seq_10()
test = seq_10(500)
train_loader = DataLoader(train, batch_size=16, trans_func=seq2label, shuffle=True)
test_loader = DataLoader(test, batch_size=64, trans_func=seq2label)
device = get_device()

from my_utils import Trainer, EvaluatorC
from torch.optim import Adam

from torch_models.models import BoV, LSTMLastHidden, LSTMMaxPool, SingleClassifier
def test_BoV():
    encoder = BoV(embed_size=50, vocab_size=10).to(device)
    model = SingleClassifier(encoder=encoder, output_size=2, hidden_size=None,
                             activation='Tanh', dropout=0, freeze_encoder=False).to(device)

    optimizer = Adam(model.parameters())

    evaluator = EvaluatorC(model, test_loader)
    trainer = Trainer(model, train_loader)
    trainer.train(optimizer, max_epoch=1,
                  evaluator=evaluator, score_monitor=None, show_log=False, hook_func=None)
    assert True == (evaluator.evaluate() > 0.9)



def test_LSTM():
    for clas in [LSTMLastHidden, LSTMMaxPool]:
        for bidirectional in [False, True]:
            for num_layers in [1, 2]:
                encoder = clas(embed_size=50, vocab_size=10, bidirectional=bidirectional, num_layers=num_layers).to(device)
                model = SingleClassifier(encoder=encoder, output_size=2, hidden_size=None,
                                         activation='Tanh', dropout=0, freeze_encoder=False).to(device)
                optimizer = Adam(model.parameters())

                evaluator = EvaluatorC(model, test_loader)
                trainer = Trainer(model, train_loader)
                trainer.train(optimizer, max_epoch=1,
                              evaluator=evaluator, score_monitor=None, show_log=False, hook_func=None)
                assert True == (evaluator.evaluate() > 0.9)
