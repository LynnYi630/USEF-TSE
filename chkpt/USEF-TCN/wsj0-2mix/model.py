"""
Library to support target speaker extraction with TCN Backend.
Oriented for low-latency edge device deployment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# === [修改 1/5]: 移除了 PositionalEncoding 的引入，因为 TCN 不需要绝对位置编码 ===

EPS = 1e-8

def select_norm(norm, dim, shape, eps=1e-8):
    if norm == "ln":
        return nn.GroupNorm(1, dim, eps=eps)
    else:
        return nn.BatchNorm1d(dim)

class FiLM(nn.Module):
    def __init__(self, size = 256):
        super(FiLM, self).__init__()
        self.linear1 = nn.Linear(size,size)
        self.linear2 = nn.Linear(size,size)
    
    def forward(self,x,aux):
        x = x * self.linear1(aux) + self.linear2(aux)
        return x

class Encoder(nn.Module):
    def __init__(self, kernel_size=2, out_channels=64, in_channels=1):
        super(Encoder, self).__init__()
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
        x = self.conv1d(x)
        x = F.relu(x)
        return x

class Decoder(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

    def forward(self, x):
        if x.dim() not in [2, 3]:
            raise RuntimeError(
                "{} accept 3/4D tensor as input".format(self.__name__)
            )
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))

        if torch.squeeze(x).dim() == 1:
            x = torch.squeeze(x, dim=1)
        else:
            x = torch.squeeze(x)
        return x


# === [修改 2/5]: 彻底删除了庞大的 Interblock (Transformer双路径) 模块 ===
# === 新增轻量级的 TCN 模块组 (GlobalLayerNorm, TCNBlock, TCNBackend) ===
class GlobalLayerNorm(nn.Module):
    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1, channel_size, 1))
        self.beta = nn.Parameter(torch.zeros(1, channel_size, 1))

    def forward(self, y):
        mean = y.mean(dim=(1, 2), keepdim=True)
        var = y.var(dim=(1, 2), keepdim=True)
        y_normalized = (y - mean) / (var + 1e-8)**0.5
        return self.gamma * y_normalized + self.beta

class TCNBlock(nn.Module):
    def __init__(self, in_channels, conv_channels, kernel_size, dilation):
        super(TCNBlock, self).__init__()
        self.conv1x1_1 = nn.Conv1d(in_channels, conv_channels, 1, bias=False)
        self.prelu1 = nn.PReLU()
        self.norm1 = GlobalLayerNorm(conv_channels)
        
        padding = (kernel_size - 1) * dilation // 2
        self.dconv = nn.Conv1d(conv_channels, conv_channels, kernel_size,
                               groups=conv_channels,
                               padding=padding,
                               dilation=dilation,
                               bias=False)
        self.prelu2 = nn.PReLU()
        self.norm2 = GlobalLayerNorm(conv_channels)
        
        self.conv1x1_2 = nn.Conv1d(conv_channels, in_channels, 1, bias=False)
        
    def forward(self, x):
        residual = x
        out = self.conv1x1_1(x)
        out = self.prelu1(out)
        out = self.norm1(out)
        out = self.dconv(out)
        out = self.prelu2(out)
        out = self.norm2(out)
        out = self.conv1x1_2(out)
        return out + residual

class TCNBackend(nn.Module):
    def __init__(self, in_channels=256, conv_channels=512, kernel_size=3, num_blocks=8, num_repeats=3):
        super(TCNBackend, self).__init__()
        layers = []
        for repeat in range(num_repeats):
            for block in range(num_blocks):
                dilation = 2 ** block
                layers.append(TCNBlock(in_channels, conv_channels, kernel_size, dilation))
        self.tcn_network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.tcn_network(x)
# =========================================================================


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
        **kwargs # 使用 kwargs 吸收掉 config 里遗留的 intra_enc, inter_enc 等旧参数，防止报错
    ):
        super(Tar_Model, self).__init__()
        self.num_spks = num_spks

        self.norm_m = select_norm("ln", in_channels, 3)
        self.conv1d1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)

        self.encoder = encoder
        self.decoder = decoder

        # === [修改 3/5]: TCN 处理的是 1D 序列，所以把原来的 Conv2d 替换为 Conv1d ===
        self.out_conv1d = nn.Conv1d(
            out_channels, out_channels * num_spks, kernel_size=1
        )
        # =========================================================================

        self.end_conv1x1 = nn.Conv1d(out_channels, out_channels, 1, bias=False)
        self.prelu = nn.PReLU()
        self.activation = nn.ReLU()
        
        self.output = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Tanh()
        )
        self.output_gate = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Sigmoid()
        )

        self.fusion_mdl = fusion_mdl
        self.fusion_norm = select_norm("ln", out_channels, 3)
        self.film = film
        
        # === [修改 4/5]: 实例化 TCN 后端，完全取代之前的 self.dual_mdl 循环列表 ===
        self.tcn_backend = TCNBackend(
            in_channels=out_channels, 
            conv_channels=512, 
            kernel_size=3, 
            num_blocks=8, 
            num_repeats=3
        )
        # =========================================================================


    def forward(self, input, aux):

        mix_w = self.encoder(input)
        aux = self.encoder(aux)
        
        x = self.norm_m(mix_w)
        aux = self.norm_m(aux)

        x = self.conv1d1(x)
        aux = self.conv1d1(aux)

        x = x.permute(0,2,1).contiguous()
        aux = aux.permute(0,2,1).contiguous()

        aux = self.fusion_mdl(x, aux)[0]
        
        x = self.film(x,aux)
        
        # 此处 x 的维度变为 [B, out_channels, L]，完美适配 TCN 的输入要求
        x = self.fusion_norm(x.permute(0,2,1).contiguous())

        # === [修改 5/5]: 删除了原有的 _Segmentation 切块机制和循环调用 ===
        # 直接将完整时序特征送入 TCN，无需再切成 3D 张量
        x = self.tcn_backend(x)
        
        x = self.prelu(x)
        
        # 使用替换后的 1D 卷积，不再需要 _over_add 进行重叠相加
        x = self.out_conv1d(x)
        
        B, _, L = x.shape
        x = x.view(B * self.num_spks, -1, L)
        # ================================================================

        x = self.output(x) * self.output_gate(x)
        x = self.end_conv1x1(x)
        _, N, L = x.shape
        x = x.view(B, self.num_spks, N, L)
        x = self.activation(x)

        x = x.transpose(0, 1)

        mix_w = torch.stack([mix_w] * self.num_spks)
        x = mix_w * x

        est_source = torch.cat(
            [
                self.decoder(x[i]).unsqueeze(-1)
                for i in range(self.num_spks)
            ],
            dim=-1,
        )

        T_origin = input.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]

        return est_source.squeeze(-1)
    
    # === [清理]: 彻底删除了原有的 _padding, _Segmentation 和 _over_add 三个成员函数 ===