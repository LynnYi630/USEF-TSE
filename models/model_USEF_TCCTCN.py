"""
Target-Conditioned Causal TCN backend for USEF-style target speaker extraction.

Design goals:
1) keep the USEF fusion interface unchanged;
2) make the temporal backend causal/streamable at the convolution level;
3) inject the frame-level target cue into every TCN block, rather than only once before the backend;
4) avoid full-sequence GlobalLayerNorm/GroupNorm in the causal backend by using frame-wise LayerNorm.

Note: the backend is causal, but the whole model is not guaranteed to be strict streaming unless
fusion_mdl and the evaluation/inference wrapper are also made chunk-wise/stateful.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


class FramewiseLayerNorm(nn.Module):
    """LayerNorm over channels independently at each frame.

    Input/Output: [B, C, T]. Unlike GroupNorm(1, C) or GlobalLayerNorm,
    this does not use statistics from future frames.
    """

    def __init__(self, channels, eps=1e-8):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        x = x.transpose(1, 2)      # [B, T, C]
        x = self.norm(x)
        return x.transpose(1, 2)   # [B, C, T]


def select_norm(norm, dim, shape=None, eps=1e-8):
    # For low-latency/causal use, prefer frame-wise LN to GroupNorm(1, C),
    # because GroupNorm would compute statistics over the time dimension.
    if norm == "ln":
        return FramewiseLayerNorm(dim, eps=eps)
    return nn.BatchNorm1d(dim)


class FiLM(nn.Module):
    def __init__(self, size=256):
        super().__init__()
        self.linear1 = nn.Linear(size, size)
        self.linear2 = nn.Linear(size, size)

    def forward(self, x, aux):
        # x, aux: [B, T, C]
        return x * self.linear1(aux) + self.linear2(aux)


class GatedResidualFiLM(nn.Module):
    """A safer FiLM variant that preserves the original feature through a residual path.

    This is optional but useful when the target cue is short/noisy: the model can decide
    how strongly to apply target conditioning at each frame/channel.
    """

    def __init__(self, size=256):
        super().__init__()
        self.gamma = nn.Linear(size, size)
        self.beta = nn.Linear(size, size)
        self.gate = nn.Linear(size * 2, size)

    def forward(self, x, aux):
        # x, aux: [B, T, C]
        gamma = torch.tanh(self.gamma(aux))
        beta = self.beta(aux)
        gate = torch.sigmoid(self.gate(torch.cat([x, aux], dim=-1)))
        return x + gate * (gamma * x + beta)


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
            x = torch.squeeze(x, dim=1)
        else:
            x = torch.squeeze(x)
        return x


class CausalDepthwiseConv1d(nn.Module):
    """Depthwise causal convolution.

    It pads only on the left, so output frame t only depends on frames <= t.
    This avoids the extra right-side computation and chomp step used in symmetric padding.
    """

    def __init__(self, channels, kernel_size, dilation=1, bias=False):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            groups=channels,
            dilation=dilation,
            padding=0,
            bias=bias,
        )

    def forward(self, x):
        if self.left_padding > 0:
            x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class TargetConditionedCausalTCNBlock(nn.Module):
    """Causal TCN block with per-block target conditioning.

    x:    [B, C, T] mixture feature
    cond: [B, C, T] frame-level target feature produced by USEF fusion

    Compared with plain CausalTCNBlock, this module keeps the target speaker cue alive
    throughout the temporal backend instead of injecting it only once before all TCN layers.
    """

    def __init__(self, in_channels, conv_channels, kernel_size, dilation, dropout=0.0):
        super().__init__()
        self.in_proj = nn.Conv1d(in_channels, conv_channels, 1, bias=False)
        self.prelu1 = nn.PReLU()
        self.norm1 = FramewiseLayerNorm(conv_channels)

        self.dconv = CausalDepthwiseConv1d(
            conv_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=False,
        )

        # gamma, beta, gate are generated from the frame-level target cue.
        self.cond_proj = nn.Conv1d(in_channels, conv_channels * 3, 1)

        self.prelu2 = nn.PReLU()
        self.norm2 = FramewiseLayerNorm(conv_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.out_proj = nn.Conv1d(conv_channels, in_channels, 1, bias=False)

    def forward(self, x, cond):
        residual = x

        out = self.in_proj(x)
        out = self.prelu1(out)
        out = self.norm1(out)

        out = self.dconv(out)

        if cond.size(-1) != out.size(-1):
            # Normally fusion_mdl should output one target cue per mixture frame.
            # This guard only prevents shape crashes when experimenting.
            cond = F.interpolate(cond, size=out.size(-1), mode="nearest")

        gamma, beta, gate = self.cond_proj(cond).chunk(3, dim=1)
        gamma = torch.tanh(gamma)
        gate = torch.sigmoid(gate)

        # Residual target modulation. The gate lets the block ignore unreliable target cues.
        out = out + gate * (gamma * out + beta)

        out = self.prelu2(out)
        out = self.norm2(out)
        out = self.dropout(out)
        out = self.out_proj(out)

        return residual + out


class TargetConditionedCausalTCNBackend(nn.Module):
    def __init__(
        self,
        in_channels=256,
        conv_channels=512,
        kernel_size=3,
        num_blocks=8,
        num_repeats=3,
        dropout=0.0,
    ):
        super().__init__()
        blocks = []
        for _ in range(num_repeats):
            for block_idx in range(num_blocks):
                dilation = 2 ** block_idx
                blocks.append(
                    TargetConditionedCausalTCNBlock(
                        in_channels=in_channels,
                        conv_channels=conv_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        dropout=dropout,
                    )
                )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, cond):
        for block in self.blocks:
            x = block(x, cond)
        return x


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
        conv_channels=512,
        kernel_size=3,
        num_blocks=8,
        num_repeats=3,
        dropout=0.0,
        use_gated_residual_film=True,
        **kwargs,
    ):
        super().__init__()
        self.num_spks = num_spks

        self.encoder = encoder
        self.decoder = decoder

        self.norm_m = select_norm("ln", in_channels, 3)
        self.conv1d1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)

        self.fusion_mdl = fusion_mdl
        self.fusion_norm = select_norm("ln", out_channels, 3)

        # Keep compatibility with the original builder: if use_gated_residual_film=False,
        # the externally passed FiLM is used.
        self.film = GatedResidualFiLM(out_channels) if use_gated_residual_film else film

        self.tcn_backend = TargetConditionedCausalTCNBackend(
            in_channels=out_channels,
            conv_channels=conv_channels,
            kernel_size=kernel_size,
            num_blocks=num_blocks,
            num_repeats=num_repeats,
            dropout=dropout,
        )

        self.prelu = nn.PReLU()
        self.out_conv1d = nn.Conv1d(out_channels, out_channels * num_spks, kernel_size=1)
        self.output = nn.Sequential(nn.Conv1d(out_channels, out_channels, 1), nn.Tanh())
        self.output_gate = nn.Sequential(nn.Conv1d(out_channels, out_channels, 1), nn.Sigmoid())
        self.end_conv1x1 = nn.Conv1d(out_channels, out_channels, 1, bias=False)
        self.activation = nn.ReLU()

    def forward(self, input, aux):
        mix_w = self.encoder(input)
        aux_w = self.encoder(aux)

        x = self.norm_m(mix_w)
        aux_feat = self.norm_m(aux_w)

        x = self.conv1d1(x)
        aux_feat = self.conv1d1(aux_feat)

        # USEF fusion expects [B, T, C].
        x_seq = x.permute(0, 2, 1).contiguous()
        aux_seq = aux_feat.permute(0, 2, 1).contiguous()

        # Frame-level target cue, usually [B, T_mix, C].
        target_seq = self.fusion_mdl(x_seq, aux_seq)[0]

        # Initial target-aware fusion before the temporal backend.
        x_seq = self.film(x_seq, target_seq)

        # Back to [B, C, T].
        x = x_seq.permute(0, 2, 1).contiguous()
        cond = target_seq.permute(0, 2, 1).contiguous()

        x = self.fusion_norm(x)
        cond = self.fusion_norm(cond)

        # Target-conditioned causal temporal modeling.
        x = self.tcn_backend(x, cond)

        x = self.prelu(x)
        x = self.out_conv1d(x)

        B, _, L = x.shape
        x = x.view(B * self.num_spks, -1, L)

        x = self.output(x) * self.output_gate(x)
        x = self.end_conv1x1(x)
        _, N, L = x.shape
        x = x.view(B, self.num_spks, N, L)
        x = self.activation(x)

        x = x.transpose(0, 1)
        mix_w = torch.stack([mix_w] * self.num_spks)
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
