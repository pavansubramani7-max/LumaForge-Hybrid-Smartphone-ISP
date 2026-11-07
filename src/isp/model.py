"""Compact residual U-Net that refines the classical ISP's display-referred RGB output."""
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(4, cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(4, cout), nn.SiLU())
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
    def forward(self, x): return self.net(x) + self.skip(x)


class TinyISPUNet(nn.Module):
    """~0.5M parameters at base=24; predicts a bounded residual for mobile export."""
    def __init__(self, in_channels=4, base=24):
        super().__init__()
        self.e1 = Block(in_channels, base)
        self.e2 = Block(base, base * 2)
        self.e3 = Block(base * 2, base * 4)
        self.down = nn.AvgPool2d(2)
        self.mid = Block(base * 4, base * 6)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.d3 = Block(base * 10, base * 4)
        self.d2 = Block(base * 6, base * 2)
        self.d1 = Block(base * 3, base)
        self.out = nn.Conv2d(base, 3, 3, padding=1)
    def forward(self, x):
        a = self.e1(x); b = self.e2(self.down(a)); c = self.e3(self.down(b))
        z = self.mid(self.down(c))
        y = self.d3(torch.cat([self.up(z), c], 1))
        y = self.d2(torch.cat([self.up(y), b], 1))
        y = self.d1(torch.cat([self.up(y), a], 1))
        return torch.clamp(x[:, :3] + 0.25 * torch.tanh(self.out(y)), 0, 1)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def infer_image(model, features_hwc, device=None, tile_size=512, overlap=64):
    """Infer an HWC frame in overlapping tiles to bound activation memory."""
    h, w = features_hwc.shape[:2]
    tile_size = max(64, int(tile_size // 8 * 8))
    overlap = min(max(0, int(overlap // 8 * 8)), tile_size // 4)
    step = max(8, tile_size - overlap)
    def starts(length):
        if length <= tile_size: return [0]
        return list(range(0, length, step))

    param = next(model.parameters())
    target_device = device if device is not None else param.device
    accum = np.zeros((h, w, 3), np.float32)
    weights = np.zeros((h, w, 1), np.float32)
    with torch.inference_mode():
        for top in starts(h):
            for left in starts(w):
                bottom, right = min(top + tile_size, h), min(left + tile_size, w)
                patch = features_hwc[top:bottom, left:right]
                ph, pw = (-patch.shape[0]) % 8, (-patch.shape[1]) % 8
                x = torch.from_numpy(patch.transpose(2, 0, 1).copy()).unsqueeze(0)
                x = x.to(device=target_device, dtype=param.dtype)
                if ph or pw: x = F.pad(x, (0, pw, 0, ph), mode="replicate")
                prediction = model(x)[..., :patch.shape[0], :patch.shape[1]]
                prediction = prediction.squeeze(0).permute(1, 2, 0).float().cpu().numpy()

                th, tw = patch.shape[:2]
                wy, wx = np.ones(th, np.float32), np.ones(tw, np.float32)
                ramp_len = min(overlap, th // 2, tw // 2)
                if top > 0 and ramp_len:
                    wy[:ramp_len] = np.linspace(.05, 1, ramp_len, dtype=np.float32)
                if bottom < h and ramp_len:
                    wy[-ramp_len:] = np.linspace(1, .05, ramp_len, dtype=np.float32)
                if left > 0 and ramp_len:
                    wx[:ramp_len] = np.linspace(.05, 1, ramp_len, dtype=np.float32)
                if right < w and ramp_len:
                    wx[-ramp_len:] = np.linspace(1, .05, ramp_len, dtype=np.float32)
                weight = (wy[:, None] * wx[None, :])[..., None]
                accum[top:bottom, left:right] += prediction * weight
                weights[top:bottom, left:right] += weight
    return accum / np.maximum(weights, 1e-8)
