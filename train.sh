export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/home/wangyi/miniconda3/envs/usef/bin/python train.py --config config/config-USEF-Mamba.yaml
# /home/wangyi/miniconda3/envs/usef/bin/python train.py --config config/config-USEF-TCN.yaml
# /home/wangyi/miniconda3/envs/usef/bin/python train.py --config config/config-USEF-SepFormer.yaml