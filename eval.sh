#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# /home/wangyi/miniconda3/envs/usef/bin/python eval.py \
#     --config chkpt/USEF-TCN/config.yaml \
#     --chkpt-path chkpt/USEF-TCN/libri2mix/temp_best.pth.tar \
#     --device 0 \
#     --memory 23 \
#     --test-set libri2mix #['wsj0-2mix', 'wham!', 'whamr!', 'libri2mix']

# /home/wangyi/miniconda3/envs/usef/bin/python eval.py \
#     --config chkpt/USEF-SepFormer/config.yaml \
#     --chkpt-path chkpt/USEF-SepFormer/libri2mix/temp_best.pth.tar \
#     --device 0 \
#     --memory 23 \
#     --test-set libri2mix #['wsj0-2mix', 'wham!', 'whamr!', 'libri2mix']

/home/wangyi/miniconda3/envs/usef/bin/python eval.py \
    --config chkpt/USEF-Mamba/config.yaml \
    --chkpt-path chkpt/USEF-Mamba/libri2mix/temp_best.pth.tar \
    --device 0 \
    --memory 23 \
    --test-set libri2mix #['wsj0-2mix', 'wham!', 'whamr!', 'libri2mix']

# /home/wangyi/miniconda3/envs/usef/bin/python eval.py \
#     --config chkpt/USEF-CausalTCN/config.yaml \
#     --chkpt-path chkpt/USEF-CausalTCN/libri2mix/temp_best.pth.tar \
#     --device 0 \
#     --memory 23 \
#     --test-set libri2mix #['wsj0-2mix', 'wham!', 'whamr!', 'libri2mix']