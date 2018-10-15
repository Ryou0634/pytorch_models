import torch

class PyTorchModelSaver():
    def __init__(self, model, save_file='./model'):
        self.model = model
        self.save_file = save_file

    def save(self, name_suffix):
        save_path = self.save_file + name_suffix + '.pth'
        torch.save(self.model.state_dict(), save_path)
