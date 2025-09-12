import torch
import torch.nn as nn
import torch.nn.functional as F

class ReAFBSReLU(nn.Module):
    """
    ReAFBSReLU activation (PyTorch) faithful to Eq. (1)-(5) in the paper.

    - Learnable per-channel parameters: tr (right threshold), tl (left threshold),
      ar (right slope), al (left slope).
    - Smoothing uses AFB (log-sum-exp) with smoothing parameter `l`.
    - Regulation: out = gamma * afbs + delta (paper uses gamma=1.25, delta=0.5).
    """

    def __init__(self,
                 num_channels: int,
                 init_tr: float = 0.4,
                 init_tl: float = -0.4,
                 init_ar: float = 0.2,
                 init_al: float = 0.2,
                 l: float = 0.05,
                 gamma: float = 1.25,
                 delta: float = 0.5,
                 learnable_l: bool = False):
        """
        Args:
            num_channels: number of channels (C) in input (N,C,H,W).
            init_tr, init_tl: initial thresholds (scalars or per-channel init).
            init_ar, init_al: initial slopes for right/left tails.
            l: smoothing parameter (recommended 0.05 in the paper).
            gamma, delta: regulation constants (paper: gamma=1.25, delta=0.5).
            learnable_l: if True, treat l as a learnable parameter (not typical).
        """
        super().__init__()
        # per-channel learnable parameters
        self.tr = nn.Parameter(torch.full((num_channels,), float(init_tr)))
        self.tl = nn.Parameter(torch.full((num_channels,), float(init_tl)))
        self.ar = nn.Parameter(torch.full((num_channels,), float(init_ar)))
        self.al = nn.Parameter(torch.full((num_channels,), float(init_al)))

        # smoothing scalar (not per-channel by default)
        if learnable_l:
            self.l = nn.Parameter(torch.tensor(float(l)))
        else:
            self.register_buffer('l', torch.tensor(float(l)))

        self.gamma = float(gamma)
        self.delta = float(delta)

    def _log_term(self, x, a, b, l):
        """
        compute ln( exp(-l*(x - a)) + exp(-l*(x + b)) )
        using torch.logaddexp for numerical stability.
        x, a, b are broadcastable tensors.
        """
        # v1 = -l * (x - a)
        # v2 = -l * (x + b)
        v1 = -l * (x - a)
        v2 = -l * (x + b)
        return torch.logaddexp(v1, v2)  # elementwise log(exp(v1)+exp(v2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (N, C, H, W) or (N, C) or (C,) etc. We expect channel dim = 1.
        Returns: same shape as x.
        """
        if x.dim() < 2:
            raise ValueError("ReAFBSReLU expects input with channel dimension (N,C,...)")

        # Ensure parameters broadcast to input shape: (1, C, 1, 1) for 4D inputs.
        # We'll handle arbitrary trailing dims by unsqueezing appropriately.
        C = x.shape[1]
        # prepare shaped params
        shape = [1, C] + [1] * (x.dim() - 2)  # e.g. (1,C,1,1) for images
        tr = self.tr.view(shape)
        tl = self.tl.view(shape)
        ar = self.ar.view(shape)
        al = self.al.view(shape)
        l = self.l if isinstance(self.l, torch.Tensor) else torch.tensor(self.l, device=x.device, dtype=x.dtype)
        if isinstance(l, torch.nn.Parameter):
            l = l.view(1, *([1] * (x.dim()-1)))
        # keep l as scalar broadcastable
        l = l.to(x.device).type_as(x)

        # Eq. (1) piecewise SReLU parts (these are used outside the smoothing interval)
        left = tl + al * (x - tl)    # for x <= tl
        middle = x                   # for tl < x < tr
        right = tr + ar * (x - tr)   # for x >= tr

        # Eq. (2) AFB log terms (we use logaddexp formulation)
        # log_term_ab = ln( exp(-l*(x - tr)) + exp(-l*(x + tl)) )
        log_term_ab = self._log_term(x, tr, tl, l)
        # log_term_tl_tl = ln( exp(-l*(x - tl)) + exp(-l*(x + tl)) )
        log_term_tl_tl = self._log_term(x, tl, tl, l)

        # Eq. (5) piecewise AFBSReLU: (I implement the same algebraic structure the paper uses)
        # (note: we follow the structure used in the author's TF snippet and the paper)
        # case1: x >= tr
        case1 = -0.5 + 0.8 * x + 0.4 * (-0.5 * (x * x - tr * tr) * log_term_ab)

        # case2: tr > x > tl (center region)
        case2 = -0.5 + 0.8 * x + 0.4 * (
            -0.5 * (x * x - tr * tr) * log_term_ab
            -0.5 * (x * x + tl * tl) * log_term_tl_tl
