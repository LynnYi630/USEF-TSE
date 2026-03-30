export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/home/wangyi/miniconda3/envs/usef/bin/python train.py --config config/config-USEF-Mamba.yaml --device 0 --memory 23
# /home/wangyi/miniconda3/envs/usef/bin/python train.py --config config/config-USEF-TCN.yaml --device 0 --memory 23
# /home/wangyi/miniconda3/envs/usef/bin/python train.py --config config/config-USEF-SepFormer.yaml --device 0 --memory 23
# /home/wangyi/miniconda3/envs/usef/bin/python train.py --config config/config-USEF-CausalTCN.yaml --device 0 --memory 23