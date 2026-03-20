#!/bin/bash

# 配置目标显存 (15GB = 15360MB)
THRESHOLD=10240
MAX=15360
TARGET_SCRIPT="$HOME/USEF-TSE/train.sh"

echo "已在当前环境执行监控...随时准备执行train.sh"

while true; do
    # 获取第 0 块显卡的显存
    VRAM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)

    if [ "$VRAM" -gt "$THRESHOLD" -a "$VRAM" -lt "$MAX" ]; then
        echo "✅ 显存达标 (${VRAM}MB)，开始执行脚本..."
        
        # 直接执行你的 eval.sh
        # 因为你已经激活了环境，这里会直接使用该环境下的 python 和工具
        bash "$TARGET_SCRIPT"
        
        echo "任务运行完毕。"
        break
    else
       sleep 1 
    fi
done
