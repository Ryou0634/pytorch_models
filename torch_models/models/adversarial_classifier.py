import numpy as np
from .functions import grad_reverse
from .classifiers import SingleClassifier

class AdversarialClassifier(SingleClassifier):
    '''
    A model for domain adaptation with gradient reversal layer.
    See "Unsupervised Domain Adaptation by Backpropagation" (Ganin, Y. and Lempitsky V., 2015)
    (http://proceedings.mlr.press/v37/ganin15.pdf)
    '''
    def __init__(self, encoder, output_size, hidden_size=0, freeze_encoder=False, gamma=10, max_epoch=50):
        super().__init__(encoder, output_size, hidden_size, freeze_encoder)
        self.gamma = gamma
        self.max_epoch = max_epoch
        self.n_epoch = 0

    def forward(self, inputs):
        encoded = self.encode(inputs)
        encoded = grad_reverse(encoded, self.get_lambd())
        if self.hidden_size > 0:
            encoded = self.fc_hidden(encoded)
        output = self.fc_out(encoded)
        return output

    def get_lambd(self):
        p = self.n_epoch/self.max_epoch
        return 2/(1+np.exp(-self.gamma*p)) - 1
