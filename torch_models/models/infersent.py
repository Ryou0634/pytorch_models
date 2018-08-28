import torch
from ._classifier import ClassifierBase


class InferSentClassifier(ClassifierBase):
    '''
    A model for natural language inference.
    See 'Supervised Learning of Universal Sentence Representations from Natural Language Inference Data'
    (Conneau, et al., 2017)
    '''
    def __init__(self, encoder, output_size, hidden_size=0,
                activation='Tanh', freeze_encoder=False):
        encoded_size = encoder.output_size*4
        super().__init__(encoder, encoded_size, output_size, hidden_size,
                         activation, freeze_encoder)

    def _encode(self, inputs):
        seq1s, seq2s = inputs
        encoded1 = self.encoder.forward(seq1s)
        encoded2 = self.encoder.forward(seq2s)
        if self.freeze_encoder:
            encoded1 = encoded1.detach()
            encoded2 = encoded2.detach()
        subtracted = torch.abs(encoded1-encoded2)
        multiplied = encoded1 * encoded2
        encoded = torch.cat([encoded1, encoded2, subtracted, multiplied], dim=1)
        return encoded
