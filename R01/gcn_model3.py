"""Generic two-head classifier; the research HGCN and decoder are withheld."""
from torch import nn


class DemoModel(nn.Module):
    def __init__(self, n_symptoms, n_acupoints, dimension=32):
        super().__init__()
        self.embedding = nn.Embedding(n_symptoms, dimension)
        self.main_output = nn.Linear(dimension, n_acupoints)
        self.adjunct_output = nn.Linear(dimension, n_acupoints)

    def forward(self, ids, mask):
        features = self.embedding(ids)
        mean = (features * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return {'main': self.main_output(mean), 'adjunct': self.adjunct_output(mean)}
