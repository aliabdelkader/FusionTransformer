import argparse
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn
import torch.cuda
from torch import nn
import torch.utils.data
from torchpack import distributed as dist
from torchpack.callbacks import InferenceRunner, MaxSaver
from torchpack.environ import set_run_dir
from torchpack.utils.config import configs
from torchpack.utils.logging import logger

from FusionTransformer.data.build import build_dataloader
from FusionTransformer.models.build import build_model
from FusionTransformer.common.solver.build import build_optimizer, build_scheduler
from FusionTransformer.modules.SemanticTorchpackTrainer import SemanticTorchpackTrainer
from FusionTransformer.modules.TorckpackCallbacks import MeanIoU, iouEval, accEval, WandbMaxSaver
from FusionTransformer.common.utils.torch_util import set_random_seed

import wandb

def main(cfg = None, output_dir = None) -> None:
    dist.init()

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())
    print("world size", dist.size())
    print("local rank", dist.local_rank())
    
    if dist.rank() == 0:
        wandb.login()
        run = wandb.init(project='FusionTransformer', config=cfg, group=cfg["MODEL"]["TYPE"], sync_tensorboard=True)
        
    set_run_dir(output_dir)

    configs.update(cfg)

    # seed
    if ('seed' not in configs.TRAIN) or (cfg.RNG_SEED is None):
        cfg.RNG_SEED = torch.initial_seed() % (2 ** 32 - 1)

    seed = cfg.RNG_SEED + dist.rank() * cfg.DATALOADER.NUM_WORKERS * cfg.SCHEDULER.MAX_EPOCH
    print(seed)
    set_random_seed(seed)
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    

    dataflow = {}
    dataflow["train"] = build_dataloader(cfg, mode='train', use_distributed=True)
    dataflow["val"] = build_dataloader(cfg, mode='val', use_distributed=True) if cfg.VAL.PERIOD > 0 else None


    model = build_model(cfg)[0]
    wandb.watch(model)

    model = torch.nn.parallel.DistributedDataParallel(
        model.cuda(),
        device_ids=[dist.local_rank()],
        find_unused_parameters=True)

    criterion =  nn.CrossEntropyLoss(ignore_index=0)
    optimizer = build_optimizer(cfg, model)
    # build lr scheduler
    scheduler = build_scheduler(cfg, optimizer)

    trainer = SemanticTorchpackTrainer(model=model,
                                   criterion=criterion,
                                   optimizer=optimizer,
                                   scheduler=scheduler,
                                   num_workers=cfg.DATALOADER.NUM_WORKERS,
                                   seed=seed, 
                                   cfg=cfg)
    trainer.train_with_defaults(
        dataflow=dataflow['train'],
        num_epochs=cfg.SCHEDULER.MAX_EPOCH,
        callbacks=[
            InferenceRunner(
                dataflow[split],
                callbacks=[
                    MeanIoU(name=f'MeanIoU/{split}', num_classes=cfg["MODEL"]["NUM_CLASSES"], ignore_label=0),
                    iouEval(name=f'iouEval/{split}', n_classes=cfg["MODEL"]["NUM_CLASSES"], ignore=0),
                    accEval(name=f'accEval/{split}', n_classes=cfg["MODEL"]["NUM_CLASSES"], ignore=0)

                ],
            ) for split in ['val']
        ] + [
            WandbMaxSaver('MeanIoU/val')
        ])

    wandb.finish()


if __name__ == '__main__':
    main()