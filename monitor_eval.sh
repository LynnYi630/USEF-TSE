#!/bin/bash

# 配置目标显存 (MB)
THRESHOLD=20480
TARGET_SCRIPT="$HOME/usef/eval.sh"
TARGET_DEVICE=0

echo "已在当前环境执行监控..."
echo "目标设备: ${TARGET_DEVICE}号卡, 目标显存阈值: ${THRESHOLD}MB"
echo "目标脚本: ${TARGET_SCRIPT}"

while true; do
    # 获取第 0 块显卡的显存
    VRAM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $TARGET_DEVICE)

    if [ "$VRAM" -gt "$THRESHOLD" ]; then
        echo "✅ 显存达标 (${VRAM}MB)，开始执行脚本..."
        
        # 直接执行你的 eval.sh
        # 因为你已经激活了环境，这里会直接使用该环境下的 python 和工具
        bash "$TARGET_SCRIPT"
        
        echo "任务运行完毕。"
        break
    else
       sleep 10 
    fi
done
