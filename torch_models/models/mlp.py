import torch
import torch.nn as nn

class MLP(nn.Module):
    '''
    Multi layer perceptrons.

    Attributes
    ----------
    fc_{i} (i >= 0) : nn.Layer
        Hidden layers.
    fc_out : nn.layer
        Output layer.
    self.criterion : nn.CrossEntropyLoss()
        Loss function.
    self.activation : nn.modules.activation
        Activation function. Default Tanh().
    '''

    def __init__(self, dims, activation='Tanh', dropout=0):
        '''
        Parameters
        ----------

        dims : List[int]
            Specify the dimensions of each layer.
        activation : str
            Activtion function.
        '''
        super().__init__()
        self.n_hidden = len(dims) - 2
        for i in range(self.n_hidden):
            setattr(self, 'fc_{}'.format(i), nn.Linear(dims[i], dims[i+1]))
        self.fc_out = nn.Linear(dims[-2], dims[-1])
        self.dropout = nn.Dropout(p=dropout)

        self.criterion = nn.CrossEntropyLoss()
        self.activation = eval('nn.{}()'.format(activation))

    def forward(self, inputs):
        for i in range(self.n_hidden):
            layer = getattr(self, 'fc_{}'.format(i))
            inputs = self.activation(layer(inputs))
            inputs = self.dropout(inputs)
        output = self.fc_out(inputs)
        return output

    def fit(self, inputs, labels, optimizer):
        if optimizer:
            self.train()
        else:
            self.eval()

        self.zero_grad()
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, labels)

        if optimizer:
            loss.backward()
            optimizer.step()
        return loss.item()

    def predict(self, inputs):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(inputs)
            _, idx = outputs.max(1)
        return idx
