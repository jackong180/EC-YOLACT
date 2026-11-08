"""Task-learned gated response attention; not an identified causal effect.

The historical module name is retained for manuscript compatibility.
No interventions, causal graph, or counterfactual estimation are implemented.
"""
import torch
from torch import nn

class CausalAttention(nn.Module):
    def __init__(self, channels, reduction=16, normalization='sum', eps=1e-6):
        super().__init__()
        if normalization not in ('sum', 'mean'):
            raise ValueError('normalization must be sum or mean')
        hidden = max(1, channels // reduction)
        self.causal_gate = nn.Sequential(nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True), nn.Conv2d(hidden, 1, 1), nn.Sigmoid())
        self.gamma = nn.Parameter(torch.zeros(1))
        self.normalization, self.eps = normalization, eps

    def forward(self, x):
        gate = self.causal_gate(x)
        z = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        weight = gate.to(z.dtype) * z.abs().mean(dim=1, keepdim=True)
        denom = (weight.sum if self.normalization == 'sum' else weight.mean)(dim=(2, 3), keepdim=True)
        weight = weight / (denom + self.eps)
        return x + self.gamma.to(x.dtype) * x * weight.to(x.dtype)
