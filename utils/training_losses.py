import torch
import torch.nn as nn
from torch.nn import functional as F

def masked_mean(x, mask):
    mask = mask.float()
    return (x * mask).sum() / (mask.sum() + 1e-6)

def compute_enhance_loss(
        Le,
        deg_type,
        color_loss,
        bright_loss,
        contrast_loss,
        ref_sim,
        omega_L,
        lambda_re=0.9,
):
    if torch.is_tensor(ref_sim):
        if ref_sim.dim() == 0:
            Lre = 1.0 - ref_sim
        else:
            Lre = (1.0 - ref_sim).mean()
    else:
        Lre = 1.0 - ref_sim

    color_mask = (deg_type < 3).float()
    light_mask = (deg_type == 3).float()
    haze_mask = (deg_type == 4).float()

    LA1 = masked_mean(color_loss, color_mask)
    LA2 = masked_mean(bright_loss, light_mask)
    LA3 = masked_mean(contrast_loss, haze_mask)

    total_loss = (
        Le
        + lambda_re * Lre
        + omega_L[0] * LA1
        + omega_L[1] * LA2
        + omega_L[2] * LA3
    )

    return total_loss, {
        "total": total_loss,
        "Le": Le,
        "Lre": Lre,
        "LA1": LA1,
        "LA2": LA2,
        "LA3": LA3,
        "omega_color": omega_L[0],
        "omega_light": omega_L[1],
        "omega_haze": omega_L[2],
    }

class L_brightness(nn.Module):
    def __init__(self):
        super(L_brightness, self).__init__()

    def forward(self, enhanced, original):
        mu_g = enhanced.mean(dim=1, keepdim=True).mean(dim=[2, 3]).squeeze(1)
        mu_og = original.mean(dim=1, keepdim=True).mean(dim=[2, 3]).squeeze(1)
        la2 = F.relu(0.5 - mu_g) + 0.5 * torch.abs(mu_g - mu_og)
        return la2

class L_contrast(nn.Module):
    def __init__(self):
        super(L_contrast, self).__init__()

    def forward(self, enhanced):
        laplacian_kernel = torch.tensor([[0, 1, 0],
                                         [1, -4, 1],
                                         [0, 1, 0]],
                                        dtype=torch.float32).view(1, 1, 3, 3).to(enhanced.device)
        gray = enhanced.mean(dim=1, keepdim=True)
        grad = F.conv2d(gray, laplacian_kernel, padding=1)
        la3 = -torch.abs(grad).mean(dim=[1, 2, 3])
        return la3

class ColorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, enhanced):
        c = enhanced.mean(dim=[2, 3])
        cr, cg, cb = c[:, 0], c[:, 1], c[:, 2]
        la1 = torch.sqrt((cr - 0.5).pow(4) +
                         (cg - 0.5).pow(4) +
                         (cb - 0.5).pow(4) + 1e-12)
        return la1
