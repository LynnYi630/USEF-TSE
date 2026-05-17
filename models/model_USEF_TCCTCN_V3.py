"""
Target-conditioned TCN V3 for USEF-style target speaker extraction.

Compared with TCCTCN-V2, this version changes the temporal backend and mask
head in four ways:
  1. residual plus skip aggregation across TCN blocks;
  2. a skip-based mask head instead of the older gated output head;
  3. configurable bounded mask activation, sigmoid by default;
  4. target conditioning on the depthwise, residual, and skip paths.

The backend can be causal at the convolution level, but the full model is not
strictly streaming while fusion_mdl remains full-sequence cross-attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FramewiseLayerNorm(nn.Module):
    """LayerNorm over channels independently at each frame.

    Input/output shape: [B, C, T].
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


def select_norm(norm, channels, eps=1e-8):
    norm = (norm or "fln").lower()
    if norm in {"fln", "frame_ln", "framewise_ln", "cln", "cumulative_ln", "ln"}:
        return FramewiseLayerNorm(channels, eps=eps)
    if norm in {"gln", "global_ln"}:
        return GlobalLayerNorm(channels, eps=eps)
    if norm in {"gn", "groupnorm"}:
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


class GatedResidualFiLM(nn.Module):
    def __init__(self, size=256):
        super().__init__()
        self.gamma = nn.Linear(size, size)
        self.beta = nn.Linear(size, size)
        self.gate = nn.Linear(size * 2, size)

    def forward(self, x, aux):
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
            return torch.squeeze(x, dim=1)
        return torch.squeeze(x)


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, channels, kernel_size, dilation=1, causal=True, bias=False):
        super().__init__()
        self.causal = bool(causal)
        self.left_padding = (kernel_size - 1) * dilation
        padding = 0 if self.causal else self.left_padding // 2
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


class TargetConditionedTCNBlockV3(nn.Module):
    def __init__(
        self,
        in_channels,
        conv_channels,
        skip_channels,
        kernel_size,
        dilation,
        norm="fln",
        causal=True,
        dropout=0.0,
        cond_scale_init=1.0,
        cond_scale_trainable=True,
        zero_init_cond_proj=False,
        condition_depthwise=True,
        condition_residual=True,
        condition_skip=True,
    ):
        super().__init__()
        self.condition_depthwise = bool(condition_depthwise)
        self.condition_residual = bool(condition_residual)
        self.condition_skip = bool(condition_skip)

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

        self.depth_cond_proj = nn.Conv1d(in_channels, conv_channels * 3, 1)
        self.residual_cond_proj = nn.Conv1d(in_channels, in_channels * 3, 1)
        self.skip_cond_proj = nn.Conv1d(in_channels, skip_channels * 3, 1)
        if zero_init_cond_proj:
            self._zero_conditioning_projections()

        cond_scale = torch.tensor(float(cond_scale_init))
        if cond_scale_trainable:
            self.cond_scale = nn.Parameter(cond_scale)
        else:
            self.register_buffer("cond_scale", cond_scale)

        self.prelu2 = nn.PReLU()
        self.norm2 = select_norm(norm, conv_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.res_proj = nn.Conv1d(conv_channels, in_channels, 1, bias=False)
        self.skip_proj = nn.Conv1d(conv_channels, skip_channels, 1, bias=False)

    def _zero_conditioning_projections(self):
        for proj in (
            self.depth_cond_proj,
            self.residual_cond_proj,
            self.skip_cond_proj,
        ):
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    def _apply_condition(self, x, cond, proj):
        if cond.size(-1) != x.size(-1):
            cond = F.interpolate(cond, size=x.size(-1), mode="nearest")

        gamma, beta, gate = proj(cond).chunk(3, dim=1)
        gamma = torch.tanh(gamma)
        gate = torch.sigmoid(gate)
        scale = torch.tanh(self.cond_scale)
        return x + scale * gate * (gamma * x + beta)

    def forward(self, x, cond):
        residual = x

        y = self.in_proj(x)
        y = self.prelu1(y)
        y = self.norm1(y)
        y = self.dconv(y)

        if self.condition_depthwise:
            y = self._apply_condition(y, cond, self.depth_cond_proj)

        y = self.prelu2(y)
        y = self.norm2(y)
        y = self.dropout(y)

        residual_update = self.res_proj(y)
        if self.condition_residual:
            residual_update = self._apply_condition(
                residual_update, cond, self.residual_cond_proj
            )

        skip = self.skip_proj(y)
        if self.condition_skip:
            skip = self._apply_condition(skip, cond, self.skip_cond_proj)

        return residual + residual_update, skip


class TargetConditionedTCNBackendV3(nn.Module):
    def __init__(
        self,
        in_channels=256,
        conv_channels=512,
        skip_channels=256,
        kernel_size=3,
        num_blocks=8,
        num_repeats=3,
        norm="fln",
        causal=True,
        dropout=0.0,
        cond_scale_init=1.0,
        cond_scale_trainable=True,
        zero_init_cond_proj=False,
        condition_depthwise=True,
        condition_residual=True,
        condition_skip=True,
    ):
        super().__init__()
        blocks = []
        for _ in range(num_repeats):
            for block_idx in range(num_blocks):
                blocks.append(
                    TargetConditionedTCNBlockV3(
                        in_channels=in_channels,
                        conv_channels=conv_channels,
                        skip_channels=skip_channels,
                        kernel_size=kernel_size,
                        dilation=2 ** block_idx,
                        norm=norm,
                        causal=causal,
                        dropout=dropout,
                        cond_scale_init=cond_scale_init,
                        cond_scale_trainable=cond_scale_trainable,
                        zero_init_cond_proj=zero_init_cond_proj,
                        condition_depthwise=condition_depthwise,
                        condition_residual=condition_residual,
                        condition_skip=condition_skip,
                    )
                )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, cond):
        skip_sum = None
        for block in self.blocks:
            x, skip = block(x, cond)
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
        conv_channels=512,
        skip_channels=256,
        kernel_size=3,
        num_blocks=8,
        num_repeats=3,
        norm="fln",
        causal=True,
        dropout=0.0,
        mask_activation="sigmoid",
        use_gated_residual_film=True,
        cond_scale_init=1.0,
        cond_scale_trainable=True,
        zero_init_cond_proj=False,
        condition_depthwise=True,
        condition_residual=True,
        condition_skip=True,
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
        self.film = GatedResidualFiLM(out_channels) if use_gated_residual_film else film

        self.tcn_backend = TargetConditionedTCNBackendV3(
            in_channels=out_channels,
            conv_channels=conv_channels,
            skip_channels=skip_channels,
            kernel_size=kernel_size,
            num_blocks=num_blocks,
            num_repeats=num_repeats,
            norm=norm,
            causal=causal,
            dropout=dropout,
            cond_scale_init=cond_scale_init,
            cond_scale_trainable=cond_scale_trainable,
            zero_init_cond_proj=zero_init_cond_proj,
            condition_depthwise=condition_depthwise,
            condition_residual=condition_residual,
            condition_skip=condition_skip,
        )

        self.skip_prelu = nn.PReLU()
        self.skip_norm = select_norm(norm, skip_channels)
        self.mask_conv1x1 = nn.Conv1d(
            skip_channels,
            out_channels * num_spks,
            kernel_size=1,
        )

    def _apply_mask_activation(self, x):
        activation = (self.mask_activation or "none").lower()
        if activation == "sigmoid":
            return torch.sigmoid(x)
        if activation == "relu":
            return F.relu(x)
        if activation == "softplus":
            return F.softplus(x)
        if activation == "none":
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
        cond = target_seq.permute(0, 2, 1).contiguous()

        x = self.fusion_norm(x)
        cond = self.fusion_norm(cond)

        _residual, skip = self.tcn_backend(x, cond)
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
