export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/home/wangyi/miniconda3/envs/usef/bin/python train.py --config config/config-USEF-TCCTCN.yaml --dataset libri2mix --device 0 --memory 0
# /home/wangyi/miniconda3/envs/usef/bin/python train.py --config config/config-USEF-TCN-WRCD.yaml --dataset libri2mix --device 0 --memory 0
# /home/wangyi/miniconda3/envs/usef/bin/python train.py --config config/config-USEF-TCCTCN.yaml --dataset wsj0-2mix --device 0 --memory 0
# /home/wangyi/miniconda3/envs/usef/bin/python train.py --config config/config-USEF-Mamba.yaml --dataset libri2mix --device 0 --memory 0
# /home/wangyi/miniconda3/envs/usef/bin/python train.py --config config/config-USEF-TCN.yaml --dataset wsj0-2mix --device 0 --memory 0
# /home/wangyi/miniconda3/envs/usef/bin/python train.py --config config/config-USEF-SepFormer.yaml --dataset wsj0-2mix --device 0 --memory 0
# /home/wangyi/miniconda3/envs/usef/bin/python train.py --config config/config-USEF-CausalTCN.yaml --dataset libri2mix --device 0 --memory 0
