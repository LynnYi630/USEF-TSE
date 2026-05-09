#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# /home/wangyi/miniconda3/envs/usef/bin/python eval.py \
#     --config chkpt/USEF-TCN/libri2mix/config.yaml \
#     --chkpt-path chkpt/USEF-TCN/libri2mix/temp_best.pth.tar \
#     --device 0 \
#     --memory 0

python eval.py \
    --config chkpt/USEF-SepFormer/wsj0-2mix/config.yaml \
    --chkpt-path chkpt/USEF-SepFormer/wsj0-2mix/temp_best.pth.tar \
    --device 0 \
    --memory 0

# /home/wangyi/miniconda3/envs/usef/bin/python eval.py \
#     --config chkpt/USEF-Mamba/libri2mix/config.yaml \
#     --chkpt-path chkpt/USEF-Mamba/libri2mix/temp_best.pth.tar \
#     --device 0 \
#     --memory 0

# /home/wangyi/miniconda3/envs/usef/bin/python eval.py \
#     --config chkpt/USEF-CausalTCN/libri2mix/config.yaml \
#     --chkpt-path chkpt/USEF-CausalTCN/libri2mix/temp_best.pth.tar \
#     --device 0 \
#     --memory 0

# /home/wangyi/miniconda3/envs/usef/bin/python eval.py \
#   --config chkpt/USEF-TCN-WRCD/libri2mix/config.yaml \
#   --chkpt-path chkpt/USEF-TCN-WRCD/libri2mix/temp_best.pth.tar \
#   --aux-duration 1.0 \
#   --device 0 \
#   --memory 0
