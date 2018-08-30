import torch
from ._classifier import ClassifierBase

class SingleClassifier(ClassifierBase):
    '''
    Classifier for single input.
    (e.g. sentiment analysis, document classification, image classification)
    '''
    def __init__(self, encoder, output_size, hidden_size=None,
                 activation='Tanh', dropout=0, freeze_encoder=False):
        encoded_size = encoder.output_size
        super().__init__(encoder, encoded_size, output_size, hidden_size,
                         activation, dropout, freeze_encoder)

class DoubleClassifier(ClassifierBase):
    '''
    Classifier for two inputs.
    (e.g. natural language inferece, discoure relation recognition)
    '''
    def __init__(self, encoder, output_size, hidden_size=None,
                 activation='Tanh', dropout=0, freeze_encoder=False):
        encoded_size = encoder.output_size*2
        super().__init__(encoder, encoded_size, output_size, hidden_size,
                         activation, dropout, freeze_encoder)

    def _encode(self, inputs):
        seq1s, seq2s = inputs
        encoded1 = self.encoder.forward(seq1s)
        encoded2 = self.encoder.forward(seq2s)
        if self.freeze_encoder:
            encoded1 = encoded1.detach()
            encoded2 = encoded2.detach()
        encoded = torch.cat([encoded1, encoded2], dim=1)
        return encoded
