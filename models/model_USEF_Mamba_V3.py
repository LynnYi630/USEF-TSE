"""
USEF-Mamba V3: Mamba-2 backend with bidirectional + skip aggregation.

Key differences vs V2:
  - Uses Mamba2 (structured state space duality, ICML 2024) instead of Mamba-1.
    Mamba-2 uses larger d_state (128 vs 64) at lower compute cost via the
    chunked scan algorithm, and is generally faster on modern GPUs.
  - Same bidirectional + skip aggregation pattern as V2; these are about
    speech separation specifically, not the SSM block choice.
  - Same forward signature as V2 so configs/eval scripts can be reused.
  - Keeps a local normalization selector. ``cln`` routes to real cumulative
    layer norm, while ``fln`` stays frame-wise.

Mamba-2 constraint: (d_model * expand) must be divisible by headdim.
With d_model=256, expand=2, headdim=64 -> (256*2)/64 = 8 heads. OK.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba2

from models.local.normalization import (
    CumulativeLayerNorm,
    FramewiseLayerNorm,
    GlobalLayerNorm,
)


def select_norm(norm, channels, shape=None, eps=1e-8):
    norm = (norm or "fln").lower()
    if norm in {"fln", "frame_ln", "framewise_ln"}:
        return FramewiseLayerNorm(channels, eps=eps)
    if norm in {"cln", "cumulative_ln"}:
        return CumulativeLayerNorm(channels, eps=eps)
    if norm in {"gln", "global_ln"}:
        return GlobalLayerNorm(channels, eps=eps)
    if norm in {"gn", "ln", "groupnorm"}:
        return nn.GroupNorm(1, channels, eps=eps)
    if norm in {"bn", "batchnorm", "batch_norm"}:
        return nn.BatchNorm1d(channels, eps=eps)
    raise ValueError("Unsupported norm type: {}".format(norm))


class FiLM(nn.Module):
    def __init__(self, size=256):
        super().__init__()
        self.linear1 = nn.Linear(size, size)
        self.linear2 = nn.Linear(size, size)

    def forward(self, x, aux):
        return x * self.linear1(aux) + self.linear2(aux)


class Encoder(nn.Module):
    def __init__(self, kernel_size=2, out_channels=64, in_channels=1):
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            groups=1,
            bias=False,
        )
        self.in_channels = in_channels

    def forward(self, x):
        if self.in_channels == 1:
            x = torch.unsqueeze(x, dim=1)
        return F.relu(self.conv1d(x))


class Decoder(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if x.dim() not in [2, 3]:
            raise RuntimeError("Decoder expects a 2D or 3D tensor as input")
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if torch.squeeze(x).dim() == 1:
            return torch.squeeze(x, dim=1)
        return torch.squeeze(x)


class BidirectionalMamba2Layer(nn.Module):
    """One Mamba-2 block with forward + reversed pass.

    Forward and backward Mamba-2 do not share params. Their outputs are summed
    before residual add.
    """

    def __init__(self, d_model, d_state=128, d_conv=4, expand=2, headdim=64, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fwd = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
        )
        self.bwd = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # See V2: 1/sqrt(2) on (fwd+bwd) keeps post-merge variance stable.
        self._bidir_scale = 1.0 / math.sqrt(2.0)

    def forward(self, x):
        residual = x
        y = self.norm(x)
        fwd = self.fwd(y)
        bwd = torch.flip(self.bwd(torch.flip(y, dims=[1])), dims=[1])
        y = (fwd + bwd) * self._bidir_scale
        y = self.dropout(y)
        return residual + y


class Mamba2BackendV3(nn.Module):
    """Bidirectional Mamba-2 stack with TCN-style skip aggregation."""

    def __init__(
        self,
        in_channels=256,
        skip_channels=256,
        num_layers=8,
        d_state=128,
        d_conv=4,
        expand=2,
        headdim=64,
        dropout=0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            BidirectionalMamba2Layer(
                d_model=in_channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=headdim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )
        self.skip_projs = nn.ModuleList(
            nn.Linear(in_channels, skip_channels) for _ in range(num_layers)
        )
        # See V2: 1/sqrt(N) skip scaling keeps mask logits from growing with depth.
        self._skip_scale = 1.0 / math.sqrt(float(num_layers))

    def forward(self, x_btc):
        skip_sum = None
        for layer, skip_proj in zip(self.layers, self.skip_projs):
            x_btc = layer(x_btc)
            skip = skip_proj(x_btc)
            skip_sum = skip if skip_sum is None else skip_sum + skip
        return x_btc, skip_sum * self._skip_scale


class Tar_Model(nn.Module):
    """USEF target speaker extraction with Mamba-2 backend."""

    def __init__(
        self,
        encoder,
        decoder,
        fusion_mdl,
        film,
        in_channels,
        out_channels,
        num_spks=1,
        mamba_num_layers=8,
        mamba_d_state=128,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_headdim=64,
        mamba_dropout=0.0,
        skip_channels=256,
        norm="gln",
        mask_activation="sigmoid",
        **kwargs,
    ):
        super().__init__()
        self.num_spks = num_spks
        self.mask_activation = mask_activation

        self.encoder = encoder
        self.decoder = decoder

        self.norm_m = select_norm(norm, in_channels)
        self.conv1d1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)

        self.fusion_mdl = fusion_mdl
        self.fusion_norm = select_norm(norm, out_channels)
        self.film = film

        self.backend = Mamba2BackendV3(
            in_channels=out_channels,
            skip_channels=skip_channels,
            num_layers=mamba_num_layers,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            headdim=mamba_headdim,
            dropout=mamba_dropout,
        )

        self.skip_prelu = nn.PReLU()
        self.skip_norm = select_norm(norm, skip_channels)
        self.mask_conv1x1 = nn.Conv1d(skip_channels, out_channels * num_spks, 1)

    def _apply_mask_activation(self, x):
        if self.mask_activation == "sigmoid":
            return torch.sigmoid(x)
        if self.mask_activation == "relu":
            return F.relu(x)
        if self.mask_activation == "softplus":
            return F.softplus(x)
        if self.mask_activation == "none":
            return x
        raise ValueError("Unsupported mask_activation: {}".format(self.mask_activation))

    def forward(self, input, aux):
        mix_w = self.encoder(input)
        aux_w = self.encoder(aux)

        x = self.norm_m(mix_w)
        aux_feat = self.norm_m(aux_w)

        x = self.conv1d1(x)
        aux_feat = self.conv1d1(aux_feat)

        x_seq = x.permute(0, 2, 1).contiguous()
        aux_seq = aux_feat.permute(0, 2, 1).contiguous()

        target_seq = self.fusion_mdl(x_seq, aux_seq)[0]
        x_seq = self.film(x_seq, target_seq)

        x = x_seq.permute(0, 2, 1).contiguous()
        x = self.fusion_norm(x)
        x = x.permute(0, 2, 1).contiguous()

        _residual, skip = self.backend(x)

        skip = skip.permute(0, 2, 1).contiguous()
        x = self.skip_prelu(skip)
        x = self.skip_norm(x)
        x = self.mask_conv1x1(x)

        B, _, L = x.shape
        x = x.view(B, self.num_spks, -1, L)
        x = self._apply_mask_activation(x)

        mix_w = torch.stack([mix_w] * self.num_spks)
        x = x.transpose(0, 1)
        x = mix_w * x

        est_source = torch.cat(
            [self.decoder(x[i]).unsqueeze(-1) for i in range(self.num_spks)],
            dim=-1,
        )

        T_origin = input.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]

        return est_source.squeeze(-1)
