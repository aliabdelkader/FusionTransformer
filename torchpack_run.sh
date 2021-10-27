pip install -e .
wandb login
torchpack dist-run -np 2 python FusionTransformer/train.py --cfg configs/semantic_kitti/lidar.yaml --use_torchpack 1
