"""
lvs_net.py
==========
LVS-Net: a lightweight retinal vessel segmentation network (0.71 M
parameters, 2.74 MB) with focal modulation attention and feature refinement.

  encoder   3 blocks, each a 3x3 conv and a 1x1 conv concatenated, then
            max-pool then batch-norm  (24 -> 48 -> 96 channels)
  bottle    Focal Modulation Attention (FMAM) + Spatial Feature Refinement (SFRB)
  decoder   3 transposed convs, each with SFRB on both the upsampled path
            and the skip connection
  output    1x1 conv -> 1 channel -> sigmoid

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-3


class Up(nn.Module):
    """Stride-2 transposed convolution with TensorFlow 'same' alignment."""

    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, k, stride=2,
                                       padding=0, output_padding=0)

    def forward(self, x):
        return self.conv(x)[..., :x.shape[-2] * 2, :x.shape[-1] * 2]


class SFRB(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 1); self.b1 = nn.BatchNorm2d(ch, eps=EPS)
        self.c2 = nn.Conv2d(2 * ch, ch, 1); self.b2 = nn.BatchNorm2d(ch, eps=EPS)
        self.c3 = nn.Conv2d(ch, ch, 1); self.b3 = nn.BatchNorm2d(ch, eps=EPS)

    def forward(self, x):
        f = F.relu(self.b1(self.c1(x)))
        f3 = F.relu(self.b2(self.c2(torch.cat(
            [F.adaptive_max_pool2d(f, 1), F.adaptive_avg_pool2d(f, 1)], 1))))
        xa = torch.sigmoid(self.b3(self.c3(F.adaptive_avg_pool2d(x, 1))))
        return x + f3 * xa


class FMAM(nn.Module):
    def __init__(self, in_ch, filters=16, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha
        self.conv1 = nn.Conv2d(in_ch, filters, 3, padding=1)
        self.conv2 = nn.Conv2d(in_ch, filters, 1)
        self.gate = nn.Conv2d(filters, filters, 1)
        self.focal = nn.Conv2d(filters, filters, 1)

    def forward(self, x):
        c1 = F.relu(self.conv1(x))
        c2 = F.relu(self.conv2(x))
        ctx = c1 * torch.sigmoid(self.gate(F.adaptive_avg_pool2d(c2, 1)))
        mod = torch.sigmoid(self.focal(
            (F.adaptive_max_pool2d(ctx, 1) - F.adaptive_avg_pool2d(ctx, 1)) * self.alpha))
        return torch.cat([c1, (ctx * mod) ** self.gamma], 1)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, chained=False):
        super().__init__()
        self.chained = chained
        self.c3 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.c1 = nn.Conv2d(out_ch if chained else in_ch, out_ch, 1)
        self.bn = nn.BatchNorm2d(2 * out_ch, eps=EPS)

    def forward(self, x):
        a = F.relu(self.c3(x))
        b = F.relu(self.c1(a if self.chained else x))
        return self.bn(F.max_pool2d(torch.cat([a, b], 1), 2)), b


class LVSNet(nn.Module):
    def __init__(self, F_=3):
        super().__init__()
        c1, c2, c3 = F_ * 8, F_ * 16, F_ * 32
        self.e1 = EncoderBlock(3, c1, chained=True)
        self.e2 = EncoderBlock(2 * c1, c2)
        self.e3 = EncoderBlock(2 * c2, c3)
        self.drop = nn.Dropout2d(0.5)
        self.fmam = FMAM(2 * c3, 16)
        self.sfrb_b = SFRB(32)
        self.up1 = Up(2 * c3 + 32, c3)
        self.s1a, self.s1b = SFRB(c3), SFRB(2 * c2)
        self.cv1 = nn.Conv2d(c3 + 2 * c2, c3, 3, padding=1)
        self.up2 = Up(c3, c2)
        self.s2a, self.s2b = SFRB(c2), SFRB(2 * c1)
        self.cv2 = nn.Conv2d(c2 + 2 * c1, c2, 3, padding=1)
        self.up3 = Up(c2, c1)
        self.s3a, self.s3b = SFRB(c1), SFRB(c1)
        self.cv3 = nn.Conv2d(2 * c1, c1, 3, padding=1)
        self.out = nn.Conv2d(c1, 1, 1)

    def forward(self, x):
        p1, skip1 = self.e1(x)
        p2, _ = self.e2(p1)
        p3, _ = self.e3(p2)
        p3 = self.drop(p3)
        b = torch.cat([p3, self.sfrb_b(self.fmam(p3))], 1)
        d = F.relu(self.up1(b))
        d = F.relu(self.cv1(torch.cat([self.s1a(d), self.s1b(p2)], 1)))
        d = F.relu(self.up2(d))
        d = F.relu(self.cv2(torch.cat([self.s2a(d), self.s2b(p1)], 1)))
        d = F.relu(self.up3(d))
        d = F.relu(self.cv3(torch.cat([self.s3a(d), self.s3b(skip1)], 1)))
        return torch.sigmoid(self.out(d))


def dice_loss(pred, target, smooth=1e-6):
    num = 2 * (pred * target).sum() + smooth
    den = (pred ** 2).sum() + (target ** 2).sum() + smooth
    return 1 - num / den


def load_lvs_net(weights, device=None):
    """Load a checkpoint into the model and put it in eval mode."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = LVSNet().to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    return model, device
