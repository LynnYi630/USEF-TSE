"""
USEF target speaker extraction with a configurable Conv-TasNet style TCN backend.

TCN-V2 keeps the USEF encoder/fusion/FiLM interface, but replaces the minimal
TCN backend with:
  - configurable block/repeat/channel settings,
  - frame-wise normalization by default,
  - residual plus skip aggregation,
  - causal or non-causal depthwise convolution,
  - a bounded mask option for more stable early training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-8


class FramewiseLayerNorm(nn.Module):
    """LayerNorm over channels independently at each frame.

    Input/output shape: [B, C, T]. This does not use statistics from future
    frames, unlike GroupNorm(1, C) over [C, T].
    """

    def __init__(self, channels, eps=1e-8):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x.transpose(1, 2)


class GlobalLayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-8):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x_float = x.float()
        mean = x_float.mean(dim=(1, 2), keepdim=True)
        var = x_float.var(dim=(1, 2), keepdim=True, unbiased=False)
        x_norm = (x_float - mean) / torch.sqrt(var + self.eps)
        x_norm = self.gamma.float() * x_norm + self.beta.float()
        return x_norm.to(dtype)


def select_norm(norm, channels, shape=None, eps=1e-8):
    norm = (norm or "fln").lower()
    if norm in {"fln", "frame_ln", "framewise_ln", "cln", "cumulative_ln"}:
        return FramewiseLayerNorm(channels, eps=eps)
    if norm in {"gln", "global_ln"}:
        return GlobalLayerNorm(channels, eps=eps)
    if norm in {"gn", "ln"}:
        return nn.GroupNorm(1, channels, eps=eps)
    if norm in {"bn", "batchnorm"}:
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


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, channels, kernel_size, dilation=1, causal=False, bias=False):
        super().__init__()
        self.causal = bool(causal)
        self.left_padding = (kernel_size - 1) * dilation
        if self.causal:
            padding = 0
        else:
            padding = self.left_padding // 2
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            groups=channels,
            dilation=dilation,
            padding=padding,
            bias=bias,
        )

    def forward(self, x):
        if self.causal and self.left_padding > 0:
            x = F.pad(x, (self.left_padding, 0))
        return self.depthwise(x)


class TCNBlockV2(nn.Module):
    def __init__(
        self,
        in_channels,
        conv_channels,
        skip_channels,
        kernel_size,
        dilation,
        norm="fln",
        causal=False,
        dropout=0.0,
    ):
        super().__init__()
        self.in_proj = nn.Conv1d(in_channels, conv_channels, 1, bias=False)
        self.prelu1 = nn.PReLU()
        self.norm1 = select_norm(norm, conv_channels)

        self.dconv = DepthwiseSeparableConv1d(
            conv_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            causal=causal,
            bias=False,
        )
        self.prelu2 = nn.PReLU()
        self.norm2 = select_norm(norm, conv_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.res_proj = nn.Conv1d(conv_channels, in_channels, 1, bias=False)
        self.skip_proj = nn.Conv1d(conv_channels, skip_channels, 1, bias=False)

    def forward(self, x):
        residual = x
        y = self.in_proj(x)
        y = self.prelu1(y)
        y = self.norm1(y)
        y = self.dconv(y)
        y = self.prelu2(y)
        y = self.norm2(y)
        y = self.dropout(y)

        residual_out = residual + self.res_proj(y)
        skip = self.skip_proj(y)
        return residual_out, skip


class TCNBackendV2(nn.Module):
    def __init__(
        self,
        in_channels=256,
        conv_channels=512,
        skip_channels=256,
        kernel_size=3,
        num_blocks=8,
        num_repeats=3,
        norm="fln",
        causal=False,
        dropout=0.0,
    ):
        super().__init__()
        blocks = []
        for _ in range(num_repeats):
            for block_idx in range(num_blocks):
                blocks.append(
                    TCNBlockV2(
                        in_channels=in_channels,
                        conv_channels=conv_channels,
                        skip_channels=skip_channels,
                        kernel_size=kernel_size,
                        dilation=2 ** block_idx,
                        norm=norm,
                        causal=causal,
                        dropout=dropout,
                    )
                )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        skip_sum = None
        for block in self.blocks:
            x, skip = block(x)
            skip_sum = skip if skip_sum is None else skip_sum + skip
        return x, skip_sum


class Tar_Model(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        fusion_mdl,
        film,
        in_channels,
        out_channels,
        num_spks=1,
        tcn_conv_channels=512,
        tcn_skip_channels=256,
        tcn_kernel_size=3,
        tcn_num_blocks=8,
        tcn_num_repeats=3,
        tcn_norm="fln",
        tcn_causal=False,
        tcn_dropout=0.0,
        mask_activation="sigmoid",
        **kwargs,
    ):
        super().__init__()
        self.num_spks = num_spks
        self.mask_activation = mask_activation

        self.encoder = encoder
        self.decoder = decoder

        self.norm_m = select_norm(tcn_norm, in_channels)
        self.conv1d1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)

        self.fusion_mdl = fusion_mdl
        self.fusion_norm = select_norm(tcn_norm, out_channels)
        self.film = film

        self.tcn_backend = TCNBackendV2(
            in_channels=out_channels,
            conv_channels=tcn_conv_channels,
            skip_channels=tcn_skip_channels,
            kernel_size=tcn_kernel_size,
            num_blocks=tcn_num_blocks,
            num_repeats=tcn_num_repeats,
            norm=tcn_norm,
            causal=tcn_causal,
            dropout=tcn_dropout,
        )

        self.skip_prelu = nn.PReLU()
        self.skip_norm = select_norm(tcn_norm, tcn_skip_channels)
        self.mask_conv1x1 = nn.Conv1d(
            tcn_skip_channels,
            out_channels * num_spks,
            kernel_size=1,
        )

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

        _residual, skip = self.tcn_backend(x)
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
