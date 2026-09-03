"""SSIM loss and EMA for the Part1 slice pipeline.

Transplanted verbatim from `bs8414_slice_surrogate/train.py` (the 60-sim
recipe) so the loss and the weight averaging under Part1 are provably the same
code, not a re-implementation that might differ in a constant.

They live in their own module because `train_slices_part1.py` must be
byte-identical across all five slice projects, and two of them
(`bs8414_samba_mlp_surrogate`, `bs8414_fundiff_surrogate`) have no `train.py`
to import these from.
"""
import torch
import torch.nn.functional as F


def gaussian_kernel(size=11, sigma=1.5, channels=1, device="cpu"):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g.unsqueeze(0) * g.unsqueeze(1)
    return kernel.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1)


_ssim_kernel_cache = {}


def ssim_loss(pred, target, kernel_size=7, sigma=1.5):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    channels = pred.shape[1]
    key = (kernel_size, sigma, channels, pred.device)

    if key not in _ssim_kernel_cache:
        _ssim_kernel_cache[key] = gaussian_kernel(kernel_size, sigma, channels,
                                                  pred.device)
    kernel = _ssim_kernel_cache[key].to(pred.dtype)
    pad = kernel_size // 2

    mu_p = F.conv2d(pred, kernel, padding=pad, groups=channels)
    mu_t = F.conv2d(target, kernel, padding=pad, groups=channels)
    s_pp = F.conv2d(pred * pred, kernel, padding=pad, groups=channels) - mu_p * mu_p
    s_tt = F.conv2d(target * target, kernel, padding=pad, groups=channels) - mu_t * mu_t
    s_pt = F.conv2d(pred * target, kernel, padding=pad, groups=channels) - mu_p * mu_t

    ssim = ((2 * mu_p * mu_t + C1) * (2 * s_pt + C2)) / \
           ((mu_p * mu_p + mu_t * mu_t + C1) * (s_pp + s_tt + C2))
    return 1.0 - ssim.mean()


class EMA:
    """Exponential moving average of the floating-point weights."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def apply_to(self, model):
        sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for k in self.shadow:
            sd[k] = self.shadow[k].clone()
        return sd
