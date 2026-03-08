import random
import numpy as np
import torch
import torch.nn.functional as F

# All tensors expected in [0,1], shape (B,3,H,W)

def gaussian_noise(x, sigma=0.01):
    noise = torch.randn_like(x) * sigma
    return torch.clamp(x + noise, 0.0, 1.0)

def gaussian_blur(x, k=5):
    # simple blur via avgpool (fast)
    pad = k // 2
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(x_pad, kernel_size=k, stride=1)

def jpeg_sim(x, quality=50):
    # simple compression simulation: downsample -> upsample (proxy for jpeg artifacts)
    # quality lower => more loss
    scale = {90: 0.95, 70: 0.85, 50: 0.75, 30: 0.6}.get(quality, 0.75)
    B, C, H, W = x.shape
    h2, w2 = int(H * scale), int(W * scale)
    y = F.interpolate(x, size=(h2, w2), mode="bilinear", align_corners=False)
    y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
    return torch.clamp(y, 0.0, 1.0)

def benign_attack(x):
    # choose one benign transformation
    r = random.random()
    if r < 0.33:
        return gaussian_noise(x, sigma=random.choice([0.005, 0.01, 0.02]))
    elif r < 0.66:
        return gaussian_blur(x, k=random.choice([3, 5]))
    else:
        return jpeg_sim(x, quality=random.choice([90, 70, 50, 30]))

def tamper_attack(x, patch_frac=0.25):
    # Remove/overwrite a patch region (simulates occlusion/tampering)
    B, C, H, W = x.shape
    ph, pw = int(H * patch_frac), int(W * patch_frac)

    y = x.clone()
    for i in range(B):
        top = random.randint(0, H - ph)
        left = random.randint(0, W - pw)

        # overwrite with random noise patch
        patch = torch.rand((C, ph, pw), device=x.device)
        y[i, :, top:top+ph, left:left+pw] = patch

    return torch.clamp(y, 0.0, 1.0)