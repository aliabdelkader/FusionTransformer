pip install -e .
wandb login
torchpack dist-run -np 2 python FusionTransformer/train.py --cfg configs/semantic_kitti/debug.yaml --use_torchpack 1