import numpy as np
from .functions import grad_reverse
from .mlp import MLP

class GradReverseMLP(MLP):
    '''
    A model for domain adaptation with gradient reversal layer.
    Take encoded representation as input, put it through gradient reversal layer, and perform classification with MLP.
    See "Unsupervised Domain Adaptation by Backpropagation" (Ganin, Y. and Lempitsky V., 2015)
    (http://proceedings.mlr.press/v37/ganin15.pdf)

    Attributes
    ----------

    gamma : int (float) > 0 or None
        A hyper-parameter to control the growth of lambda.
        The bigger it is, the faster lambda grows.
        If None, get_lamd() always returns 1.
    max_update : int
        A hyper-parameter to control the growth of lambda.
        The bigger it is, the slower lambda grows.
    n_update : int
        The namber of
        The number of updates. It is incremented every time self.get_lambd() is called.
        The bigger it is, the closer to 1 lambda is.
    '''
    def __init__(self, dims, activation='Tanh', dropout=0, gamma=None, max_update=50):
        super().__init__(dims, activation=activation, dropout=dropout)
        self.gamma = gamma
        self.max_update = max_update
        self.n_update = 0

    def forward(self, encoded):
        encoded = grad_reverse(encoded, self.get_lambd())
        return super().forward(encoded)

    def get_lambd(self):
        p = self.n_update/self.max_update
        if self.gamma:
            return 2/(1+np.exp(-self.gamma*p)) - 1
        else:
            return 1
