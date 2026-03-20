import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

# 辅助函数：选择归一化方式
def select_norm(norm, dim, shape, eps=1e-8):
    if norm == "ln":
        return nn.GroupNorm(1, dim, eps=eps)
    else:
        return nn.BatchNorm1d(dim)

# ==========================================
# 核心创新点：为边缘实时语音提取设计的 Mamba 后端
# ==========================================
class MambaBackend(nn.Module):
    def __init__(self, in_channels=256, num_layers=6, d_state=16, d_conv=4, expand=2):
        super(MambaBackend, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            Mamba(
                d_model=in_channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            ) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(in_channels) for _ in range(num_layers)])

    def forward(self, x):
        # Mamba 需要 [Batch, Length, Channels]
        x = x.permute(0, 2, 1).contiguous()
        for layer, norm in zip(self.layers, self.norms):
            residual = x
            x = norm(x)
            x = layer(x)
            x = x + residual
        # 算完后变回 [Batch, Channels, Length]
        x = x.permute(0, 2, 1).contiguous()
        return x

# ==========================================
# 外层封装：完全对齐原版 USEF 接口的 Tar_Model
# ==========================================
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

        # ==== 严谨的前端特征处理层 ====
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
        
        # ==== 替换为 Mamba 后端 ====
        self.backend = MambaBackend(in_channels=out_channels, num_layers=6)

    def forward(self, input, aux):
        # 1. 编码器分别提取混合语音和参考声纹特征
        mix_w = self.encoder(input)
        aux = self.encoder(aux)
        
        x = self.norm_m(mix_w)
        aux = self.norm_m(aux)

        x = self.conv1d1(x)
        aux = self.conv1d1(aux)

        # 调整维度适配 Transformer 融合模块
        x = x.permute(0,2,1).contiguous()
        aux = aux.permute(0,2,1).contiguous()

        # 2. 跨模态特征融合
        aux = self.fusion_mdl(x, aux)[0]
        x = self.film(x,aux)
        
        # 还原回 1D 卷积格式
        x = self.fusion_norm(x.permute(0,2,1).contiguous())

        # 3. 送入 Mamba 进行全局长序列建模
        x = self.backend(x)
        
        # 4. 后处理与掩码生成机制
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

        # 5. 掩码相乘与解码输出
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

# ==== 保持原样的基础组件 ====
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