from torch.optim import SGD, Adam
from my_utils import Trainer, ScoreMonitor, EvaluatorC, EvaluatorLoss, EvaluatorSeq
from torch_models.utils import PyTorchModelSaver
from my_utils.misc.logging import logger


go_up_dict = {'loss': False, 'accuracy': True, 'BLEU': True}
eval_dict = {'loss': EvaluatorLoss, 'accuracy': EvaluatorC, 'BLEU': EvaluatorSeq}

def train_lr_decay(model, train_loader, dev_loader, measure,
                   checkpoint_steps=5000, max_steps=100000,
                   optimizer='Adam', initial_lr=0.1, decay_rate=0.5, lr_threshold=1e-6,
                   save_path=None):

    # Preparing
    trainer = Trainer(model, train_loader)
    dev_evaluator = eval_dict[measure](model, dev_loader)
    score_monitor = ScoreMonitor(threshold=1, go_up=go_up_dict[measure])
    model_saver = PyTorchModelSaver(model, save_file=save_path) if save_path else None

    optimizer = eval(optimizer)(model.parameters(), lr=initial_lr)

    lr = initial_lr
    logger.info('Start Training')
    while lr > lr_threshold:
        logger.info('learning rate: {}'.format(lr))
        trainer.train_step(optimizer, checkpoint_steps=checkpoint_steps, max_steps=max_steps,
                           evaluator=dev_evaluator, score_monitor=score_monitor,
                           model_saver=model_saver)
        lr *= decay_rate
        for param_group in optimizer.param_groups:
                param_group['lr'] = lr
