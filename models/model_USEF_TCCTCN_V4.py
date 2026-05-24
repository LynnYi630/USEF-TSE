"""
USEF-TCCTCN-V4: repeat-level, skip-only gated target conditioning.

V4 keeps the stable USEF-TCN-V2 backbone and moves target conditioning into a
small residual adapter on each repeat-level skip output:

    TCN repeat -> skip_r
    skip_r + scale * need_gate * reliability_gate * Adapter(skip_r, cond)

This keeps the residual temporal path unchanged and lets target cues correct
the mask-related skip pathway only when the learned gates open.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.local.normalization import (
    CumulativeLayerNorm,
    FramewiseLayerNorm,
    GlobalLayerNorm,
)

from model_USEF_TCN_V2 import TCNBlockV2


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


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, channels, kernel_size, dilation=1, causal=False, bias=False):
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


class RepeatSkipConditionAdapter(nn.Module):
    def __init__(
        self,
        skip_channels,
        cond_channels,
        norm="gln",
        bottleneck_channels=None,
        use_reliability_gate=True,
        need_gate_init_bias=0.0,
        reliability_gate_init_bias=2.0,
        zero_init_delta=True,
    ):
        super().__init__()
        self.use_reliability_gate = bool(use_reliability_gate)
        bottleneck_channels = int(bottleneck_channels or skip_channels)

        self.cond_proj = nn.Conv1d(cond_channels, skip_channels, 1, bias=False)

        self.delta_in = nn.Conv1d(skip_channels * 2, bottleneck_channels, 1)
        self.delta_prelu = nn.PReLU()
        self.delta_norm = select_norm(norm, bottleneck_channels)
        self.delta_out = nn.Conv1d(bottleneck_channels, skip_channels, 1)

        self.need_gate_in = nn.Conv1d(skip_channels * 4, skip_channels, 1)
        self.need_gate_prelu = nn.PReLU()
        self.need_gate_norm = select_norm(norm, skip_channels)
        self.need_gate_out = nn.Conv1d(skip_channels, skip_channels, 1)

        if self.use_reliability_gate:
            reliability_hidden = max(skip_channels // 2, 32)
            self.reliability_mlp = nn.Sequential(
                nn.Linear(skip_channels * 2, reliability_hidden),
                nn.PReLU(),
                nn.Linear(reliability_hidden, skip_channels),
            )
        else:
            self.reliability_mlp = None

        nn.init.constant_(self.need_gate_out.bias, float(need_gate_init_bias))
        if self.reliability_mlp is not None:
            nn.init.constant_(self.reliability_mlp[-1].bias, float(reliability_gate_init_bias))

        if zero_init_delta:
            nn.init.zeros_(self.delta_out.weight)
            nn.init.zeros_(self.delta_out.bias)

    def forward(self, skip, cond, scale):
        if cond.size(-1) != skip.size(-1):
            cond = F.interpolate(cond, size=skip.size(-1), mode="nearest")

        cond_skip = self.cond_proj(cond)

        delta = torch.cat([skip, cond_skip], dim=1)
        delta = self.delta_in(delta)
        delta = self.delta_prelu(delta)
        delta = self.delta_norm(delta)
        delta = self.delta_out(delta)

        gate_input = torch.cat(
            [skip, cond_skip, torch.abs(skip - cond_skip), skip * cond_skip],
            dim=1,
        )
        need_gate = self.need_gate_in(gate_input)
        need_gate = self.need_gate_prelu(need_gate)
        need_gate = self.need_gate_norm(need_gate)
        need_gate = torch.sigmoid(self.need_gate_out(need_gate))

        if self.reliability_mlp is None:
            reliability_gate = 1.0
        else:
            cond_float = cond_skip.float()
            cond_mean = cond_float.mean(dim=-1)
            cond_std = cond_float.std(dim=-1, unbiased=False)
            cond_stats = torch.cat([cond_mean, cond_std], dim=1)
            reliability_gate = torch.sigmoid(self.reliability_mlp(cond_stats))
            reliability_gate = reliability_gate.to(skip.dtype).unsqueeze(-1)

        return skip + scale * need_gate * reliability_gate * delta


class RepeatLevelSkipTCNBackend(nn.Module):
    def __init__(
        self,
        in_channels=256,
        conv_channels=512,
        skip_channels=256,
        kernel_size=3,
        num_blocks=8,
        num_repeats=3,
        norm="gln",
        causal=False,
        dropout=0.0,
        adapter_bottleneck_channels=None,
        adapter_scale_init_logit=-3.0,
        adapter_scale_trainable=True,
        use_reliability_gate=True,
        need_gate_init_bias=0.0,
        reliability_gate_init_bias=2.0,
        zero_init_adapter_delta=True,
    ):
        super().__init__()
        self.num_blocks = int(num_blocks)
        self.num_repeats = int(num_repeats)

        blocks = []
        adapters = []
        for repeat_idx in range(self.num_repeats):
            for block_idx in range(self.num_blocks):
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
            adapters.append(
                RepeatSkipConditionAdapter(
                    skip_channels=skip_channels,
                    cond_channels=in_channels,
                    norm=norm,
                    bottleneck_channels=adapter_bottleneck_channels,
                    use_reliability_gate=use_reliability_gate,
                    need_gate_init_bias=need_gate_init_bias,
                    reliability_gate_init_bias=reliability_gate_init_bias,
                    zero_init_delta=zero_init_adapter_delta,
                )
            )

        # Keep this flat name so TCN-V2 checkpoints can initialize the blocks.
        self.blocks = nn.ModuleList(blocks)
        self.repeat_adapters = nn.ModuleList(adapters)

        scale_logit = torch.tensor(float(adapter_scale_init_logit))
        if adapter_scale_trainable:
            self.adapter_scale_logit = nn.Parameter(scale_logit)
        else:
            self.register_buffer("adapter_scale_logit", scale_logit)

    def forward(self, x, cond):
        skip_sum = None
        block_offset = 0
        scale = torch.sigmoid(self.adapter_scale_logit)

        for repeat_idx in range(self.num_repeats):
            repeat_skip = None
            for _ in range(self.num_blocks):
                x, skip = self.blocks[block_offset](x)
                block_offset += 1
                repeat_skip = skip if repeat_skip is None else repeat_skip + skip

            repeat_skip = self.repeat_adapters[repeat_idx](repeat_skip, cond, scale)
            skip_sum = repeat_skip if skip_sum is None else skip_sum + repeat_skip

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
        tcn_norm="gln",
        tcn_causal=False,
        tcn_dropout=0.0,
        mask_activation="sigmoid",
        adapter_bottleneck_channels=None,
        adapter_scale_init_logit=-3.0,
        adapter_scale_trainable=True,
        use_reliability_gate=True,
        need_gate_init_bias=0.0,
        reliability_gate_init_bias=2.0,
        zero_init_adapter_delta=True,
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

        self.tcn_backend = RepeatLevelSkipTCNBackend(
            in_channels=out_channels,
            conv_channels=tcn_conv_channels,
            skip_channels=tcn_skip_channels,
            kernel_size=tcn_kernel_size,
            num_blocks=tcn_num_blocks,
            num_repeats=tcn_num_repeats,
            norm=tcn_norm,
            causal=tcn_causal,
            dropout=tcn_dropout,
            adapter_bottleneck_channels=adapter_bottleneck_channels,
            adapter_scale_init_logit=adapter_scale_init_logit,
            adapter_scale_trainable=adapter_scale_trainable,
            use_reliability_gate=use_reliability_gate,
            need_gate_init_bias=need_gate_init_bias,
            reliability_gate_init_bias=reliability_gate_init_bias,
            zero_init_adapter_delta=zero_init_adapter_delta,
        )

        self.skip_prelu = nn.PReLU()
        self.skip_norm = select_norm(tcn_norm, tcn_skip_channels)
        self.mask_conv1x1 = nn.Conv1d(
            tcn_skip_channels,
            out_channels * num_spks,
            kernel_size=1,
        )

    def freeze_backbone(self, freeze=True):
        """Freeze (or unfreeze) the TCN-V2 trunk inside tcn_backend.

        Only the unconditioned TCNBlockV2 stack is touched. The repeat-level
        adapters and adapter_scale_logit remain trainable, so the model can
        still learn the V4 conditioning path on top of a fixed backbone.
        Returns the count of parameters whose requires_grad changed.
        """
        changed = 0
        for param in self.tcn_backend.blocks.parameters():
            if param.requires_grad == (not freeze):
                param.requires_grad = not freeze
                changed += param.numel()
        return changed

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
