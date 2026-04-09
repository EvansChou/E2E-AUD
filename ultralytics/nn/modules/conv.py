# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convolution modules."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = (
    "Conv",
    "Conv2",
    "HWD",
    "LDConv",
    "UnitEnhance",
    "UnitModule",
    "LightConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "RepConv",
    "Index",
    "DSConv"
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization."""
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse

class DSConv(nn.Module):
    """The Basic Depthwise Separable Convolution."""
    def __init__(self, c_in, c_out, k=3, s=1, p=None, d=1, bias=False):
        super().__init__()
        if p is None:
            p = (d * (k - 1)) // 2
        self.dw = nn.Conv2d(
            c_in, c_in, kernel_size=k, stride=s,
            padding=p, dilation=d, groups=c_in, bias=bias
        )
        self.pw = nn.Conv2d(c_in, c_out, 1, 1, 0, bias=bias)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return self.act(self.bn(x))


class HWD(nn.Module):
    """
    Haar Wavelet Downsampling (HWD).

    A stride-2 downsampling module that decomposes features into 4 Haar subbands (LL, HL, LH, HH),
    concatenates them along channels (4*C), then uses a 1x1 Conv+BN+SiLU for representation learning.

    Note: This implementation is dependency-free (does not require `pytorch_wavelets`).
    """

    default_act = nn.SiLU()

    def __init__(self, c1, c2, act=True):
        super().__init__()
        self.c1 = c1

        # Fixed 2x2 Haar filters. Applied per-channel via grouped conv.
        # (LL, HL, LH, HH) order matches common HWD implementations.
        w = torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],  # LL
                [[1.0, -1.0], [1.0, -1.0]],  # HL
                [[1.0, 1.0], [-1.0, -1.0]],  # LH
                [[1.0, -1.0], [-1.0, 1.0]],  # HH
            ],
            dtype=torch.float32,
        ) * 0.5
        weight = w[:, None, :, :].repeat(c1, 1, 1, 1)  # (4*c1, 1, 2, 2)
        self.register_buffer("_haar_weight", weight, persistent=False)

        self.conv1x1 = nn.Conv2d(4 * c1, c2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        # Zero-pad to even spatial size to avoid dropping the last row/col for odd inputs.
        h, w = x.shape[-2:]
        if (h & 1) or (w & 1):
            x = F.pad(x, (0, w & 1, 0, h & 1), mode="constant", value=0.0)
        y = F.conv2d(x, self._haar_weight.to(dtype=x.dtype), stride=2, padding=0, groups=self.c1)
        y = self.conv1x1(y)
        return self.act(self.bn(y))


class UnitEnhance(nn.Module):
    """
    UnitEnhance: a lightweight joint image enhancement module (minimal, detection-loss-only variant).

    This is a minimal integration-friendly version of UnitModule:
    - Predicts a 3-channel transmission map t(x) in (0, 1) from a lightweight backbone.
    - Uses per-image atmospheric light A as channel-wise spatial mean.
    - Enhances input image with a Koschmieder-style formula: (I - (1 - t) * A) / t.

    Notes:
    - Designed to be inserted before the detector as a plug-and-play module.
    - No extra unsupervised losses are implemented in this minimal version.
    - Expects input images scaled to [0, 1] float (Ultralytics training pipeline already does this).
    """

    def __init__(
        self,
        c1: int,
        stem_channels=(16, 32),
        lk_channels=32,
        large_kernels=(7, 7),
        small_kernel=3,
        t_min: float = 1e-3,
    ):
        super().__init__()
        assert c1 == 3, "UnitEnhance is intended for RGB inputs (3 channels)."
        assert len(stem_channels) == 2
        assert len(large_kernels) == 2
        self.t_min = float(t_min)

        c_s1, c_s2 = stem_channels
        self.stem = nn.Sequential(
            Conv(c1, c_s1, k=3, s=2, p=1),
            Conv(c_s1, c_s2, k=3, s=2, p=1),
        )

        # Lightweight large-kernel depthwise blocks (RepLKNet-style simplified).
        self.lk1 = self._lk_block(c_s2, lk_channels, large_kernels[0], small_kernel)
        self.lk2 = self._lk_block(c_s2, lk_channels, large_kernels[1], small_kernel)

        # Transmission head: upsample x2 -> conv -> upsample x2 -> conv -> sigmoid.
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.t_conv1 = Conv(c_s2, c_s2, k=3, s=1, p=1)
        self.t_conv2 = nn.Conv2d(c_s2, c1, kernel_size=3, stride=1, padding=1, bias=True)

    @staticmethod
    def _lk_block(c: int, dw_ratio: int, large_k: int, small_k: int) -> nn.Module:
        dw_c = int(dw_ratio)
        return nn.Sequential(
            Conv(c, dw_c, k=1, s=1, p=0),
            nn.Sequential(
                nn.Conv2d(dw_c, dw_c, kernel_size=large_k, stride=1, padding=large_k // 2, groups=dw_c, bias=False),
                nn.BatchNorm2d(dw_c),
                nn.SiLU(),
            ),
            nn.Sequential(
                nn.Conv2d(dw_c, dw_c, kernel_size=small_k, stride=1, padding=small_k // 2, groups=dw_c, bias=False),
                nn.BatchNorm2d(dw_c),
                nn.SiLU(),
            ),
            Conv(dw_c, c, k=1, s=1, p=0, act=False),
        )

    def forward(self, x):
        # x: (B, 3, H, W), expected in [0, 1]
        f = self.stem(x)
        y = f + self.lk1(f)
        y = y + self.lk2(y)

        t = self.t_conv1(self.up1(y))
        t = self.t_conv2(self.up2(t))
        t = torch.sigmoid(t)
        t = torch.clamp(t, min=self.t_min, max=1.0)

        a = torch.mean(x, dim=(-2, -1), keepdim=True)  # (B, 3, 1, 1)
        x_enh = self.denoise(x, t, a)
        return torch.clamp(x_enh, 0.0, 1.0)

    @staticmethod
    def noise(x, t, a):
        """Noise image (forward Koschmieder model)."""
        return x * t + (1.0 - t) * a

    @staticmethod
    def denoise(x, t, a):
        """Denoise image (inverse Koschmieder model)."""
        return (x - (1.0 - t) * a) / t


class UnitModule(UnitEnhance):
    """
    Full UnitModule-style joint enhancement module.

    Compared to UnitEnhance, UnitModule additionally computes unsupervised auxiliary losses during training
    (as described in the UnitModule paper) and exposes them via `self.unit_losses` for joint optimization
    with detection loss.
    """

    def __init__(
        self,
        c1: int,
        stem_channels=(16, 32),
        lk_channels=32,
        large_kernels=(7, 7),
        small_kernel=3,
        alpha: float = 0.9,
        t_min: float = 1e-3,
        loss_t_weight: float = 1.0,
        loss_acc_weight: float = 0.1,
        loss_cc_weight: float = 0.1,
        loss_sp_weight: float = 0.1,
        loss_tv_weight: float = 0.01,
    ):
        super().__init__(
            c1=c1,
            stem_channels=stem_channels,
            lk_channels=lk_channels,
            large_kernels=large_kernels,
            small_kernel=small_kernel,
            t_min=t_min,
        )
        assert 0.0 < alpha < 1.0
        self.alpha = float(alpha)

        # Loss weights
        self.loss_t_weight = float(loss_t_weight)
        self.loss_acc_weight = float(loss_acc_weight)
        self.loss_cc_weight = float(loss_cc_weight)
        self.loss_sp_weight = float(loss_sp_weight)
        self.loss_tv_weight = float(loss_tv_weight)

        # Lightweight color-cast predictor head (ACC loss), using pooled features.
        self.acc_down = nn.Conv2d(stem_channels[-1], 3, kernel_size=1, stride=1, padding=0, bias=True)
        self.acc_head = nn.Sequential(
            nn.Linear(49, 32),
            nn.SiLU(),
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )

        self.unit_loss = None
        self.unit_loss_dict = {}

    @staticmethod
    def _mse(a, b):
        return (a - b).pow(2).mean()

    @staticmethod
    def _saturated_pixel_loss(a, b):
        zero = a.new_zeros(1)
        one = a.new_ones(1)
        loss_max = (torch.max(a, one) + torch.max(b, one) - 2 * one).nanmean()
        loss_min = -(torch.min(a, zero) + torch.min(b, zero)).nanmean()
        return loss_max + loss_min

    @staticmethod
    def _total_variation_loss(x):
        _, _, h, w = x.shape
        h_tv = (x[:, :, 1:, :] - x[:, :, : h - 1, :]).pow(2).mean()
        w_tv = (x[:, :, :, 1:] - x[:, :, :, : w - 1]).pow(2).mean()
        return h_tv + w_tv

    @staticmethod
    def _color_cast_loss(x):
        # Encourage channel means to be balanced (underwater color cast reduction).
        m = torch.mean(x, dim=(-2, -1))
        return (m - m[:, [1, 2, 0]]).pow(2).mean()

    def _assisting_color_cast_loss(self, feature, a):
        # Approximate the paper's RoIPool-based ACC with adaptive pooling to 7x7.
        b = feature.shape[0]
        pooled = F.adaptive_avg_pool2d(feature, (7, 7))
        pooled = self.acc_down(pooled).view(b, 3, -1)  # (b, 3, 49)
        pred = self.acc_head(pooled).squeeze(-1)  # (b, 3)
        a_vec = a.squeeze(-1).squeeze(-1)  # (b, 3)
        return self._mse(pred, a_vec)

    def forward(self, x):
        # Clear previous losses to avoid stale reads.
        self.unit_loss = None
        self.unit_loss_dict = {}

        # Forward (same as UnitEnhance) but keep intermediate feature for ACC loss.
        f = self.stem(x)
        y = f + self.lk1(f)
        y = y + self.lk2(y)

        t = self.t_conv1(self.up1(y))
        t = self.t_conv2(self.up2(t))
        t = torch.sigmoid(t)
        t = torch.clamp(t, min=self.t_min, max=1.0)

        a = torch.mean(x, dim=(-2, -1), keepdim=True)  # (B, 3, 1, 1)
        x_denoise = self.denoise(x, t, a)

        if self.training:
            # Create fake degraded image and predict its t/A, then enforce transmission consistency.
            x_fake = self.noise(x, self.alpha, a)
            # Predict t_fake and a_fake (a_fake is still mean-based).
            f_fake = self.stem(x_fake)
            y_fake = f_fake + self.lk1(f_fake)
            y_fake = y_fake + self.lk2(y_fake)
            t_fake = self.t_conv1(self.up1(y_fake))
            t_fake = self.t_conv2(self.up2(t_fake))
            t_fake = torch.sigmoid(t_fake)
            t_fake = torch.clamp(t_fake, min=self.t_min, max=1.0)
            a_fake = torch.mean(x_fake, dim=(-2, -1), keepdim=True)
            x_fake_denoise = self.denoise(x_fake, t_fake, a_fake)

            loss_t = self._mse(self.alpha * t, t_fake) * self.loss_t_weight
            self.unit_loss_dict["unit/loss_t"] = loss_t.detach()

            if self.loss_acc_weight > 0:
                loss_acc = self._assisting_color_cast_loss(y, a) * self.loss_acc_weight
                self.unit_loss_dict["unit/loss_acc"] = loss_acc.detach()

            if self.loss_cc_weight > 0:
                loss_cc = self._color_cast_loss(x_denoise) * self.loss_cc_weight
                self.unit_loss_dict["unit/loss_cc"] = loss_cc.detach()

            if self.loss_sp_weight > 0:
                loss_sp = self._saturated_pixel_loss(x_denoise, x_fake_denoise) * self.loss_sp_weight
                self.unit_loss_dict["unit/loss_sp"] = loss_sp.detach()

            if self.loss_tv_weight > 0:
                loss_tv = self._total_variation_loss(x_denoise) * self.loss_tv_weight
                self.unit_loss_dict["unit/loss_tv"] = loss_tv.detach()

            # Sum for joint training.
            if self.unit_loss_dict:
                self.unit_loss = sum(self.unit_loss_dict.values())

        x_denoise = torch.clamp(x_denoise, 0.0, 1.0)
        return x_denoise


class LDConv(nn.Module):
    """
    Linear Deformable Convolution with arbitrary parameter count and initial sampled shape.

    Args:
        c_in (int): Number of input channels.
        c_out (int): Number of output channels.
        num_param (int): Number of sampled points in the kernel.
        stride (int, optional): Sampling stride for offset generation and downsampling. Defaults to 1.
        bias (bool, optional): Whether to use bias in the aggregation conv. Defaults to False.
    """

    def __init__(self, c_in, c_out, num_param, stride=1, bias=False):
        super().__init__()
        self.num_param = num_param
        self.stride = stride
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias),
            nn.BatchNorm2d(c_out),
            nn.SiLU(),
        )
        self.p_conv = nn.Conv2d(c_in, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.register_buffer("p_n", self._get_p_n(num_param))

    def forward(self, x):
        offset = self.p_conv(x)
        # Guard against NaN/Inf in offsets to avoid invalid gather indices.
        offset = torch.nan_to_num(offset, nan=0.0, posinf=0.0, neginf=0.0)
        dtype = offset.dtype
        N = offset.size(1) // 2
        p = self._get_p(offset, dtype)
        p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)

        p = p.permute(0, 2, 3, 1).contiguous()
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat(
            [
                torch.clamp(q_lt[..., :N], 0, x.size(2) - 1),
                torch.clamp(q_lt[..., N:], 0, x.size(3) - 1),
            ],
            dim=-1,
        ).long()
        q_rb = torch.cat(
            [
                torch.clamp(q_rb[..., :N], 0, x.size(2) - 1),
                torch.clamp(q_rb[..., N:], 0, x.size(3) - 1),
            ],
            dim=-1,
        ).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        p = torch.cat(
            [
                torch.clamp(p[..., :N], 0, x.size(2) - 1),
                torch.clamp(p[..., N:], 0, x.size(3) - 1),
            ],
            dim=-1,
        )

        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        x_offset = (
            g_lt.unsqueeze(dim=1) * x_q_lt
            + g_rb.unsqueeze(dim=1) * x_q_rb
            + g_lb.unsqueeze(dim=1) * x_q_lb
            + g_rt.unsqueeze(dim=1) * x_q_rt
        )

        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        return self.conv(x_offset)

    def _get_p_n(self, N):
        base_int = round(math.sqrt(self.num_param))
        row_number = self.num_param // base_int
        mod_number = self.num_param % base_int
        p_n_x, p_n_y = torch.meshgrid(torch.arange(0, row_number), torch.arange(0, base_int))
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)
        if mod_number > 0:
            mod_p_n_x, mod_p_n_y = torch.meshgrid(torch.arange(row_number, row_number + 1), torch.arange(0, mod_number))
            p_n_x = torch.cat((p_n_x, torch.flatten(mod_p_n_x)))
            p_n_y = torch.cat((p_n_y, torch.flatten(mod_p_n_y)))
        p_n = torch.cat([p_n_x, p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).float()
        return p_n

    def _get_p_0(self, h, w, N, device, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride, device=device),
            torch.arange(0, w * self.stride, self.stride, device=device),
        )
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        return torch.cat([p_0_x, p_0_y], 1).type(dtype)

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        p_0 = self._get_p_0(h, w, N, offset.device, dtype)
        return p_0 + self.p_n.to(dtype).to(offset.device) + offset

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x_flat = x.contiguous().view(b, c, -1)
        index = q[..., :N] * padded_w + q[..., N:]
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        # Extra safety: clamp indices to valid range to prevent CUDA gather out-of-bounds.
        index = index.clamp_(0, x_flat.size(-1) - 1)
        x_offset = x_flat.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        b, c, h, w, n = x_offset.size()
        return x_offset.permute(0, 1, 2, 4, 3).reshape(b, c, h * n, w)

class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes Ghost Convolution module with primary and cheap operations for efficient feature learning."""
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)


class Index(nn.Module):
    """Returns a particular index of the input."""

    def __init__(self, c1, c2, index=0):
        """Returns a particular index of the input."""
        super().__init__()
        self.index = index

    def forward(self, x):
        """
        Forward pass.

        Expects a list of tensors as input.
        """
        return x[self.index]
