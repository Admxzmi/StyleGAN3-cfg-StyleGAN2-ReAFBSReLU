import torch
import torch.nn as nn
import numpy as np
import dnnlib

# -----------------------------
# Base AFBSReLU function
# -----------------------------
def AFBSReLU(x, tr=1.0, tl=1.0):
    def log_term(a, b):
        # Equivalent to tf.math.logaddexp
        return torch.logaddexp(-0.05 * (x - a), -0.05 * (x + b))

    case1 = -0.5 + 0.8 * x + 0.4 * (-0.5 * (x**2 - tr**2) * log_term(tr, tl))
    case2 = -0.5 + 0.8 * x + 0.4 * (
        -0.5 * (x**2 - tr**2) * log_term(tr, tl)
        - 0.5 * (x**2 + tl**2) * log_term(tl, tl)
    )
    case3 = -0.5 + 0.8 * x - 0.4 * (0.5 * (x**2 + tl**2) * log_term(tr, tl))

    return torch.where(x >= 0.4, case1, torch.where(x > -0.4, case2, case3))


# -----------------------------
# Final Proposed ReAFBSReLU
# -----------------------------
class ReAFBSReLU(nn.Module):
    def __init__(self, tr=1.0, tl=1.0):
        super(ReAFBSReLU, self).__init__()
        self.tr = tr
        self.tl = tl

    def forward(self, x):
        afbs = AFBSReLU(x, self.tr, self.tl)
        return 1.25 * afbs + 0.5


# -----------------------------
# Wrap for StyleGAN3 activation system
# -----------------------------
def reafbsrelu(x, alpha=0, **kwargs):
    # alpha is just a placeholder here to stay consistent with StyleGAN3
    return ReAFBSReLU()(x)
