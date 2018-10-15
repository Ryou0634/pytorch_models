# torch_models

This is a repository where I personally save machine learning models, most of which are based on pytorch.

### Note
These models are not all fully tested. It is very likely the codes contain bugs. Be careful when you use the code.

## Description

Typically, models have the following methods.  
These methods are necessary to use the modules in my_utils such as Trainer and Evaluator.

```python3
def forward(self, inputs):
    '''
    Perform forward computation.
    '''
    return output

def fit(self, inputs, labels, optimizer):
    '''
    Caliculate loss and update parameters.
    '''
    return loss_item

def predict(self, inputs):
    '''
    Outputs predicted labels (or values) from inputs.
    '''
    return predicted
```

## Usage Example

```python3
>>> train, test = get_dataset()

>>> from my_utils import get_device
>>> device = get_device()
===== Device =====
cpu

>>> from my_utils import get_device, DataLoader, torch_stack
>>> train_loader = DataLoader(train, batch_size=64, trans_func=torch_stack)
>>> test_loader = DataLoader(test, batch_size=64, trans_func=torch_stack)

>>> from torch_models import MLP
>>> model = MLP([784, 50, 10])
>>> print(model)
MLP(
  (fc_0): Linear(in_features=784, out_features=50, bias=True)
  (fc_out): Linear(in_features=50, out_features=10, bias=True)
  (criterion): CrossEntropyLoss()
  (activation): Tanh()
)

>>> from torch.optim import SGD
>>> from my_utils import Trainer, EvaluatorC

>>> optimizer = SGD(model.parameters(), lr=0.1)
>>> trainer = Trainer(model, train_loader)
>>> evaluator = EvaluatorC(model, test_loader)
>>> trainer.train_epoch(optimizer, max_epoch=10, evaluator=evaluator, show_log=True)
epoch 0  	loss: 0.1835	accuracy: 0.9491
epoch 1  	loss: 0.1595	accuracy: 0.9544
epoch 2  	loss: 0.1428	accuracy: 0.956
epoch 3  	loss: 0.1298	accuracy: 0.9598
epoch 4  	loss: 0.1188	accuracy: 0.9629
epoch 5  	loss: 0.1094	accuracy: 0.964
epoch 6  	loss: 0.1022	accuracy: 0.9663
epoch 7  	loss: 0.09551	accuracy: 0.9665
epoch 8  	loss: 0.08942	accuracy: 0.968
epoch 9  	loss: 0.08407	accuracy: 0.9706
```
