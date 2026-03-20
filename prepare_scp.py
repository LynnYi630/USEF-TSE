import os
import glob

# Libri2Mix 8k min 版本的根目录
base_data_dir = "/home/wangyi/datasets/Libri2Mix/wav8k/min"

# 我们需要处理的三个数据子集
subsets = ["train-100", "dev", "test"]

for subset in subsets:
    # 动态拼接当前子集的输入路径和输出路径
    data_dir = os.path.join(base_data_dir, subset)
    out_dir = f"./data_scp/{subset}"  # 将生成的scp文件统一存放在项目下的 data_scp 目录
    
    os.makedirs(out_dir, exist_ok=True)

    # 注意：这里改成了 mix_both，让模型学习在带噪环境中提取
    mix_files = sorted(glob.glob(os.path.join(data_dir, "mix_both", "*.wav")))
    s1_files = sorted(glob.glob(os.path.join(data_dir, "s1", "*.wav")))

    # 如果这个子集没找到数据，就跳过并提示
    if not mix_files:
        print(f"警告：在 {data_dir} 中没有找到数据，请检查路径。")
        continue

    # 写入 scp 文件
    with open(os.path.join(out_dir, "mix.scp"), "w") as f_mix, \
         open(os.path.join(out_dir, "ref.scp"), "w") as f_ref, \
         open(os.path.join(out_dir, "aux.scp"), "w") as f_aux:
        
        for mix_p, s1_p in zip(mix_files, s1_files):
            utt_id = os.path.basename(mix_p).replace(".wav", "")
            
            f_mix.write(f"{utt_id} {mix_p}\n")
            f_ref.write(f"{utt_id} {s1_p}\n")
            f_aux.write(f"{utt_id} {s1_p}\n") 

    print(f"成功生成 {subset} 集合的 SCP 文件！存放在: {out_dir}")