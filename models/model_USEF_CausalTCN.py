"""
Library to support target speaker extraction with Causal TCN Backend.
Oriented for strictly zero look-ahead, low-latency edge device deployment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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

# ==========================================
# 新增：因果截断模块
# 作用：将 1D 卷积在右侧（未来时间）产生的多余 padding 截断
# ==========================================
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # 截掉序列最后面的 chomp_size 个时间步
        if self.chomp_size > 0:
            return x[:, :, :-self.chomp_size].contiguous()
        return x

# ==========================================
# 改造：严格因果的 TCN 模块组
# ==========================================
class CausalTCNBlock(nn.Module):
    def __init__(self, in_channels, conv_channels, kernel_size, dilation):
        super(CausalTCNBlock, self).__init__()
        self.conv1x1_1 = nn.Conv1d(in_channels, conv_channels, 1, bias=False)
        self.prelu1 = nn.PReLU()
        
        # 替换 GlobalLayerNorm 为 GroupNorm(1, C)，保证严格因果
        self.norm1 = nn.GroupNorm(1, conv_channels)
        
        # 核心改造：非对称 Padding
        padding = (kernel_size - 1) * dilation
        self.dconv = nn.Conv1d(conv_channels, conv_channels, kernel_size,
                               groups=conv_channels,
                               padding=padding, 
                               dilation=dilation,
                               bias=False)
        # 通过 Chomp1d 把右边对称加出来的未来部分砍掉
        self.chomp = Chomp1d(padding) 
        
        self.prelu2 = nn.PReLU()
        self.norm2 = nn.GroupNorm(1, conv_channels)
        
        self.conv1x1_2 = nn.Conv1d(conv_channels, in_channels, 1, bias=False)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1x1_1(x)
        out = self.prelu1(out)
        out = self.norm1(out)
        
        out = self.dconv(out)
        out = self.chomp(out) 
        
        out = self.prelu2(out)
        out = self.norm2(out)
        out = self.conv1x1_2(out)
        
        return out + residual

class CausalTCNBackend(nn.Module):
    def __init__(self, in_channels=256, conv_channels=512, kernel_size=3, num_blocks=8, num_repeats=3):
        super(CausalTCNBackend, self).__init__()
        layers = []
        for repeat in range(num_repeats):
            for block in range(num_blocks):
                dilation = 2 ** block
                layers.append(CausalTCNBlock(in_channels, conv_channels, kernel_size, dilation))
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
        **kwargs
    ):
        super(Tar_Model, self).__init__()
        self.num_spks = num_spks

        self.norm_m = select_norm("ln", in_channels, 3)
        self.conv1d1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)

        self.encoder = encoder
        self.decoder = decoder

        self.out_conv1d = nn.Conv1d(
            out_channels, out_channels * num_spks, kernel_size=1
        )

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
        
        # === 实例化严格因果 TCN 后端 ===
        self.tcn_backend = CausalTCNBackend(
            in_channels=out_channels, 
            conv_channels=512, 
            kernel_size=3, 
            num_blocks=8, 
            num_repeats=3
        )

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
        
        x = self.fusion_norm(x.permute(0,2,1).contiguous())

        x = self.tcn_backend(x)
        
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