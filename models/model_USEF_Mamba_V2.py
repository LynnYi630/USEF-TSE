"""
USEF-Mamba (V2): bidirectional + skip-aggregated Mamba backend.

Key fixes over the previous USEF-Mamba implementation:
  1. Bidirectional Mamba: forward Mamba + backward Mamba per layer (sum-fused),
     so the model can use both past and future context (offline mode).
  2. Skip aggregation: every layer projects its output to a skip space and
     accumulates, mirroring Conv-TasNet/TCN's information bottleneck design.
  3. Cleaner mask head: skip_sum -> PReLU -> norm -> 1x1 -> sigmoid.
     The old `tanh*sigmoid` gating after a redundant `out_conv1d` was removed.
  4. FiLM is applied once before the backend (matching TCN-V2 baseline). Per-layer
     conditioning can be added later via adapters; this V2 keeps things minimal
     to first establish a clean Mamba baseline.

This module mirrors the TCN-V2 backbone interface so configs and trainer
plumbing can be reused.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm.modules.mamba_simple import Mamba

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


class BidirectionalMambaLayer(nn.Module):
    """One Mamba block consisting of forward + reversed Mamba.

    Inputs are [B, T, C] (Mamba's native layout). The two Mambas share neither
    parameters nor states; their outputs are summed before the residual add,
    keeping the parameter count at 2x of a unidirectional layer.
    """

    def __init__(self, d_model, d_state=64, d_conv=4, expand=2, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fwd_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.bwd_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # fwd/bwd are approximately independent; raw sum doubles variance and
        # inflates gradients into each Mamba's dt. Scaling by 1/sqrt(2) keeps
        # post-merge variance ~ sigma^2 (DPMamba uses *0.5 for similar reason).
        self._bidir_scale = 1.0 / math.sqrt(2.0)

    def forward(self, x):
        residual = x
        y = self.norm(x)
        fwd = self.fwd_mamba(y)
        bwd = torch.flip(self.bwd_mamba(torch.flip(y, dims=[1])), dims=[1])
        y = (fwd + bwd) * self._bidir_scale
        y = self.dropout(y)
        return residual + y


class MambaBackendV2(nn.Module):
    """Bidirectional Mamba stack with TCN-style skip aggregation.

    The forward signature matches TCNBackendV2: returns (residual_out, skip_sum).
    skip_sum is the sum over all layers of a 1x1 projection of that layer's
    output into skip space. Following Conv-TasNet, skip_sum is what feeds the
    mask head — using only the residual stream loses too much information.
    """

    def __init__(
        self,
        in_channels=256,
        skip_channels=256,
        num_layers=8,
        d_state=64,
        d_conv=4,
        expand=2,
        dropout=0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            BidirectionalMambaLayer(
                d_model=in_channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )
        # 1x1 projections from each layer output to skip space.
        # We use Linear (not Conv1d) so we can stay in [B, T, C] layout inside
        # the backend and avoid extra permutes per layer.
        self.skip_projs = nn.ModuleList(
            nn.Linear(in_channels, skip_channels) for _ in range(num_layers)
        )
        # Sum of N independent skip vectors has variance ~ N*sigma^2; scaling by
        # 1/sqrt(N) restores the per-skip variance so mask logits don't grow with
        # depth. Without this, sigmoid easily saturates at depth >= ~12.
        self._skip_scale = 1.0 / math.sqrt(float(num_layers))

    def forward(self, x_btc):
        """x_btc: [B, T, C]. Returns (residual [B, T, C], skip_sum [B, T, skip])."""
        skip_sum = None
        for layer, skip_proj in zip(self.layers, self.skip_projs):
            x_btc = layer(x_btc)
            skip = skip_proj(x_btc)
            skip_sum = skip if skip_sum is None else skip_sum + skip
        return x_btc, skip_sum * self._skip_scale


class Tar_Model(nn.Module):
    """USEF target speaker extraction with bidirectional Mamba backend."""

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
        mamba_d_state=64,
        mamba_d_conv=4,
        mamba_expand=2,
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

        self.backend = MambaBackendV2(
            in_channels=out_channels,
            skip_channels=skip_channels,
            num_layers=mamba_num_layers,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
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

        # Backend works in [B, T, C].
        # We still apply norm in [B, C, T] to match TCN-V2's behavior.
        x = x_seq.permute(0, 2, 1).contiguous()
        x = self.fusion_norm(x)
        x = x.permute(0, 2, 1).contiguous()

        _residual, skip = self.backend(x)

        # Back to [B, C, T] for the mask head.
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
