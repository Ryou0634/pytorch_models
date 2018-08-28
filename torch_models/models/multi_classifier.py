import torch.nn as nn

class MultiClassifier(nn.Module):
    def __init__(self, models, weights):
        super().__init__()
        for i, model in enumerate(models):
            setattr(self, 'model{}'.format(i), model)
        self.weights = weights

        self.mode = 0

    def forward(self, inputs):
        model = getattr(self, 'model{}'.format(self.mode))
        return model.forward(inputs)

    def predict(self, inputs):
        model = getattr(self, 'model{}'.format(self.mode))
        return model.predict(inputs)

    def fit(self, multi_inputs, multi_labels, optimizer):
        self.zero_grad()
        total_loss = 0
        for i, (inputs, labels) in enumerate(zip(multi_inputs, multi_labels)):
            model = getattr(self, 'model{}'.format(i))
            outputs = model.forward(inputs)
            total_loss += self.weights[i]*model.criterion(outputs, labels)
        total_loss.backward()
        optimizer.step()
        return total_loss.item()
