import torch

# 1. 加载你刚刚训练好的 SepFormer 最优权重
old_ckpt_path = 'chkpt/USEF-SepFormer/libri2mix/temp_best.pth.tar'
old_ckpt = torch.load(old_ckpt_path, map_location='cpu', weights_only=False)

new_model_state_dict = {}

# 2. 遍历原有的权重字典，进行安全过滤
for k, v in old_ckpt['model_state_dict'].items():
    # 过滤掉原本 SepFormer 庞大的 Interblock (dual_mdl) 后端模块权重
    # 还要过滤掉分类层的 2D 卷积，因为我们在 TCN 里改成了 1D 卷积
    if 'dual_mdl' in k or 'conv2d' in k:
        continue
    # 其余的前端权重（如 encoder, decoder, fusion_mdl 等）全盘接收
    new_model_state_dict[k] = v

# 3. 构建一个新的存档字典
# 清空优化器状态（因为 TCN 后端是随机初始化的，动量需要重新积累），重置 epoch
new_ckpt = {
    'epoch': 0,
    'model_state_dict': new_model_state_dict,
    'optimizer_state_dict': None, 
    'best_val_loss': float('inf')
}

# 4. 保存为供 TCN 读取的初始化权重
torch.save(new_ckpt, 'tcn_init.pth.tar')
print(f"权重迁移成功！保留了 {len(new_model_state_dict)} 个有效张量，已生成 tcn_init.pth.tar")