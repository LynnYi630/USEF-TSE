import os
import random
from pathlib import Path

def generate_tse_scp():
    # 你的纯净语音库根路径 (已更名为 wsj0)
    wsj0_dir = Path("/mnt/d/Datasets/wsj0").resolve()
    # 混合数据集路径
    wsj0_2mix_dir = Path("/mnt/d/Datasets/wsj0-2mix/2speakers/wav8k/min").resolve()
    # 输出 SCP 文件的目录
    output_dir = Path("/mnt/d/Datasets/wsj0-2mix_scp")

    # 只需要从中寻找 aux 的三个核心子目录
    target_subdirs = ["si_tr_s", "si_dt_05", "si_et_05"]

    print("正在精准扫描指定的纯净语音库，构建说话人声纹字典...")
    spk_to_wavs = {}
    
    # 仅遍历需要的三个文件夹
    for sub_dir in target_subdirs:
        current_dir = wsj0_dir / sub_dir
        if not current_dir.exists():
            print(f"警告: 找不到目录 {current_dir}，请检查路径。")
            continue
            
        for wav_file in current_dir.rglob("*.wav"):
            spk_id = wav_file.name[:3]  
            if spk_id not in spk_to_wavs:
                spk_to_wavs[spk_id] = []
            spk_to_wavs[spk_id].append(wav_file)

    splits = ['tr', 'cv', 'tt']
    for split in splits:
        split_out_dir = output_dir / split
        split_out_dir.mkdir(parents=True, exist_ok=True)

        mix_dir = wsj0_2mix_dir / split / 'mix'
        s1_dir = wsj0_2mix_dir / split / 's1'
        s2_dir = wsj0_2mix_dir / split / 's2'

        if not mix_dir.exists():
            continue

        with open(split_out_dir / "mix.scp", "w", encoding="utf-8") as mix_scp, \
             open(split_out_dir / "ref.scp", "w", encoding="utf-8") as ref_scp, \
             open(split_out_dir / "aux.scp", "w", encoding="utf-8") as aux_scp:

            mix_files = sorted(list(mix_dir.glob("*.wav")))

            for mix_wav in mix_files:
                base_name = mix_wav.stem
                parts = base_name.split('_')
                if len(parts) < 4:
                    continue

                s1_part = parts[0]
                s2_part = parts[2]
                s1_id = s1_part[:3]
                s2_id = s2_part[:3]

                # 随机挑选除了混合语音本身之外的纯净音频作为 aux
                def get_aux(spk_id, exclude_part):
                    candidates = [w for w in spk_to_wavs.get(spk_id, []) if exclude_part not in w.name]
                    return random.choice(candidates) if candidates else None

                aux1_wav = get_aux(s1_id, s1_part)
                aux2_wav = get_aux(s2_id, s2_part)

                if not aux1_wav or not aux2_wav:
                    continue

                # 生成说话人 1 作为提取目标的标签对
                utt_id_1 = f"{base_name}_1"
                mix_scp.write(f"{utt_id_1} {mix_wav.absolute()}\n")
                ref_scp.write(f"{utt_id_1} {(s1_dir / mix_wav.name).absolute()}\n")
                aux_scp.write(f"{utt_id_1} {aux1_wav.absolute()}\n")

                # 生成说话人 2 作为提取目标的标签对
                utt_id_2 = f"{base_name}_2"
                mix_scp.write(f"{utt_id_2} {mix_wav.absolute()}\n")
                ref_scp.write(f"{utt_id_2} {(s2_dir / mix_wav.name).absolute()}\n")
                aux_scp.write(f"{utt_id_2} {aux2_wav.absolute()}\n")

        print(f"[{split}] 集 SCP 生成完毕，共 {len(mix_files) * 2} 条提取样本。")

if __name__ == "__main__":
    generate_tse_scp()
