"""Energy-aware Attention (EGA) for EC-YOLACT.

For each image independently:
    E[b, 1, i, j] = sum_c X[b, c, i, j] ** 2
    A = E / (sum_{i,j} E + eps)
    Y = X + gamma * X * A

The manuscript calls the learnable residual coefficient beta. Its state-dict
name remains ``gamma`` for compatibility with the reviewed checkpoints.
No entropy, softmax, or logarithm is used. Input/output: [B, C, H, W].
"""
import math
import torch
from torch import nn

__all__ = ["EGA", "EnergyAttention"]


class EGA(nn.Module):
    """Shape-preserving spatial energy attention.

    Args:
        eps: Positive denominator stabilizer.
        initial_scale: Initial residual coefficient; 0 gives identity mapping.

    One learnable scalar is shared across channels and spatial positions.
    Attention normalization is per image, never across the batch dimension.
    ``gamma`` is unconstrained; a larger energy need not imply a true defect.
    """

    def __init__(self, eps: float = 1e-6, initial_scale: float = 0.0):
        super().__init__()
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and greater than zero")
        if not math.isfinite(initial_scale):
            raise ValueError("initial_scale must be finite")
        self.eps = float(eps)
        self.gamma = nn.Parameter(torch.tensor([float(initial_scale)]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("EGA expects [batch, channels, height, width]")
        if not x.is_floating_point():
            raise TypeError("EGA expects a floating-point feature tensor")

        # Use FP32 for half/bfloat16 energy reductions. Keep float64 unchanged.
        features = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        energy = features.square().sum(dim=1, keepdim=True)
        attention = energy / (energy.sum(dim=(2, 3), keepdim=True) + self.eps)

        # [B,1,H,W] broadcasts to [B,C,H,W]; no explicit repeat is necessary.
        return x + self.gamma.to(x.dtype) * x * attention.to(x.dtype)


# Backward-compatible import used by earlier reviewed code and notebooks.
EnergyAttention = EGA


if __name__ == "__main__":
    module = EGA()
    feature = torch.randn(2, 256, 68, 68, requires_grad=True)
    output = module(feature)
    output.square().mean().backward()
    print("input:", tuple(feature.shape), "output:", tuple(output.shape))
    print("identity at initialization:", torch.equal(output, feature))
    print("gamma gradient is finite:", bool(torch.isfinite(module.gamma.grad).all()))
