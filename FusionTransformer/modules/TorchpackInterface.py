import torch
import torch.backends.cudnn
import torch.cuda
from torch import nn
import torch.utils.data
from torchpack import distributed as dist
from torchpack.callbacks import InferenceRunner
from torchpack.environ import set_run_dir
from torchpack.utils.config import configs

from FusionTransformer.data.build import build_dataloader
from FusionTransformer.models.build import build_model
from FusionTransformer.common.solver.build import build_optimizer, build_scheduler
from FusionTransformer.modules.SemanticTorchpackTrainer import SemanticTorchpackTrainer
from FusionTransformer.modules.TorchpackCallbacks import MeanIoU, iouEval, accEval, WandbMaxSaver
from FusionTransformer.common.utils.torch_util import set_random_seed

import wandb

def create_callbacks(callback_name: str = "", num_classes: int = 1, ignore_label: int = 0, output_tensor: str = ""):
    return [
        MeanIoU(name='MeanIoU/'+ callback_name, num_classes=num_classes, ignore_label=ignore_label, output_tensor=output_tensor),
        iouEval(name='iouEval/'+ callback_name, n_classes=num_classes, ignore=ignore_label, output_tensor=output_tensor),
        accEval(name='accEval/'+ callback_name, n_classes=num_classes, ignore=ignore_label, output_tensor=output_tensor),
        WandbMaxSaver('MeanIoU/'+ callback_name)
    ]


def main(cfg = None, output_dir = None) -> None:
    dist.init()

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())
    print("world size", dist.size())
    print("local rank", dist.local_rank())
    
    if dist.rank() == 0:
        run = wandb.init(project='FusionTransformer', config=cfg, group=cfg["MODEL"]["TYPE"], sync_tensorboard=True)

    set_run_dir(output_dir)

    configs.update(cfg)

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
    dist.barrier()
    if dist.rank() == 0:
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
    call_backs = []
    if cfg.MODEL.USE_FUSION == True:
        call_backs += create_callbacks(callback_name= "val/image", num_classes=cfg["MODEL"]["NUM_CLASSES"], ignore_label= 0, output_tensor="img_seg_logit")
        call_backs += create_callbacks(callback_name= "val/lidar", num_classes=cfg["MODEL"]["NUM_CLASSES"], ignore_label= 0, output_tensor="lidar_seg_logit")
    elif cfg.MODEL.USE_LIDAR == True:
        call_backs += create_callbacks(callback_name= "val/lidar", num_classes=cfg["MODEL"]["NUM_CLASSES"], ignore_label= 0, output_tensor="lidar_seg_logit")
    elif cfg.MODEL.USE_IMAGE == True:
        call_backs += create_callbacks(callback_name= "val/image", num_classes=cfg["MODEL"]["NUM_CLASSES"], ignore_label= 0, output_tensor="img_seg_logit")

    trainer.train_with_defaults(
        dataflow=dataflow['train'],
        num_epochs=cfg.SCHEDULER.MAX_EPOCH,
        callbacks=[
            InferenceRunner(
                dataflow['val'],
                callbacks=call_backs)
        ]
    )

    dist.barrier()
    if dist.rank() == 0:
        wandb.finish()


if __name__ == '__main__':
    main()