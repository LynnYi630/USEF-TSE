import torch
from mamba_ssm import Mamba

batch, length, dim = 2, 64, 256
x = torch.randn(batch, length, dim).to("cuda")
model = Mamba(d_model=dim).to("cuda")
y = model(x)
print(f"Mamba 输出形状: {y.shape}")
