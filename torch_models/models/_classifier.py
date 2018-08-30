import torch
import torch.nn as nn
import torch.nn.functional as F
from .mlp import MLP

class ClassifierBase(nn.Module):
    '''
    Classifier with encoder.
    Encoder is supposed to take some kind of input (e.g. word seqences, image pixels)
    and outputs a fixed-length representation.
    This classifier use fully connected layers (up to two layers) to classify the inputs.

    Attributes
    ----------
    encoder : Encoder
        Encoder which takes input and returns fixed-size representaion.
    mlp : MLP
        Muti layer perceptron to perform classification.
    freeze_encoder : bool
        If True, the parameters in encoder will not be updated during training.
    '''
    def __init__(self, encoder, encoded_size, output_size, hidden_size,
                 activation, dropout, freeze_encoder):
        super().__init__()
        self.encoder = encoder
        self.mlp = MLP(dims=self._get_dims(encoded_size, output_size, hidden_size),
                       activation=activation, dropout=dropout)
        self.freeze_encoder = freeze_encoder

    def _get_dims(self, encoded_size, output_size, hidden_size):
        if isinstance(hidden_size, int):
            dims = [encoded_size, hidden_size, output_size]
        elif isinstance(hidden_size, list):
            dims = [encoded_size] + hidden_size + [output_size]
        elif hidden_size is None:
            dims = [encoded_size, output_size]
        else:
            raise TypeError("hidden_size must be int or list. None if no hidden layer.")
        return dims

    def _encode(self, inputs):
        encoded = self.encoder.forward(inputs)
        if self.freeze_encoder:
            encoded = encoded.detach()
        return encoded

    def forward(self, inputs):
        encoded = self._encode(inputs)
        output = self.mlp.forward(encoded)
        return output

    def fit(self, inputs, labels, optimizer):
        self.zero_grad()
        encoded = self._encode(inputs)
        loss_item = self.mlp.fit(encoded, labels, optimizer)
        return loss_item

    def predict(self, inputs):
        with torch.no_grad():
            encoded = self._encode(inputs)
            idx = self.mlp.predict(encoded)
        return idx
