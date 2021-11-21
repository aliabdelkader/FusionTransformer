import torch
import torch.backends.cudnn
import torch.cuda
from torch import nn
import torch.utils.data
from torchpack import distributed as dist
from torchpack.callbacks import InferenceRunner
from torchpack.environ import set_run_dir
from torchpack.utils.config import configs
from torchpack.callbacks import (ConsoleWriter, EstimatedTimeLeft,
                         JSONLWriter, MetaInfoSaver, ProgressBar)
from FusionTransformer.data.build import build_dataloader
from FusionTransformer.models.build import build_model
from FusionTransformer.common.solver.build import build_optimizer, build_scheduler
from FusionTransformer.modules.SemanticTorchpackTrainer import SemanticTorchpackTrainer
from torchpack.callbacks import Callbacks
from FusionTransformer.modules.TorchpackCallbacks import MeanIoU, iouEval, accEval, WandbMaxSaver, TFEventWriterExtended, SaverRestoreIOU
from FusionTransformer.common.utils.torch_util import set_random_seed

import wandb
import os
import tqdm
def create_callbacks(callback_name: str = "", num_classes: int = 1, ignore_label: int = 0, output_tensor: str = ""):
    return [
        MeanIoU(name='MeanIoU/'+ callback_name, num_classes=num_classes, ignore_label=ignore_label, output_tensor=output_tensor),
        # iouEval(name='iouEval/'+ callback_name, n_classes=num_classes, ignore=ignore_label, output_tensor=output_tensor),
        accEval(name='accEval/'+ callback_name, n_classes=num_classes, ignore=ignore_label, output_tensor=output_tensor),
    ]

def create_saver(callback_name: str = ""):
    return [WandbMaxSaver('MeanIoU/'+ callback_name)]


def main(cfg = None, output_dir = None, run_name = "") -> None:
    os.environ[
        "TORCH_DISTRIBUTED_DEBUG"
    ] = "DETAIL"  # set to DETAIL for runtime logging
    
    dist.init()

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())
    print("world size", dist.size())
    print("local rank", dist.local_rank())
    
    if dist.rank() == 0:
        run = wandb.init(project='FusionTransformer', name=run_name, config=cfg, group=cfg["MODEL"]["TYPE"], sync_tensorboard=True)

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
    dataflow["val"] = build_dataloader(cfg, mode='val', use_distributed=True)
    dataflow["test"] = build_dataloader(cfg, mode='test', use_distributed=True)


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
    inference_callbacks = []
    if cfg.MODEL.USE_FUSION:
        inference_callbacks += create_callbacks(callback_name="val/image", num_classes=cfg["MODEL"]["NUM_CLASSES"], ignore_label=0, output_tensor="img_seg")
        inference_callbacks += create_callbacks(callback_name="val/lidar", num_classes=cfg["MODEL"]["NUM_CLASSES"], ignore_label=0, output_tensor="lidar_seg")
        test_inference_callbacks = [MeanIoU(name='MeanIoU/test/lidar', num_classes=cfg["MODEL"]["NUM_CLASSES"], ignore_label=0, output_tensor="lidar_seg")]

    elif cfg.MODEL.USE_LIDAR:
        inference_callbacks += create_callbacks(callback_name= "val/lidar", num_classes=cfg["MODEL"]["NUM_CLASSES"], ignore_label=0, output_tensor="lidar_seg")
        test_inference_callbacks = [MeanIoU(name='MeanIoU/test/lidar', num_classes=cfg["MODEL"]["NUM_CLASSES"], ignore_label=0, output_tensor="lidar_seg")]
    
    elif cfg.MODEL.USE_IMAGE:
        inference_callbacks += create_callbacks(callback_name= "val/image", num_classes=cfg["MODEL"]["NUM_CLASSES"], ignore_label=0, output_tensor="img_seg")
        test_inference_callbacks = [MeanIoU(name='MeanIoU/test/lidar', num_classes=cfg["MODEL"]["NUM_CLASSES"], ignore_label=0, output_tensor="img_seg")]
    
    saver_callbacks = []
    if cfg.MODEL.SAVE:
        if cfg.MODEL.USE_FUSION:
            saver_callbacks += create_saver(callback_name="val/image")
            saver_callbacks += create_saver(callback_name="val/lidar")
            
        elif cfg.MODEL.USE_LIDAR:
            saver_callbacks += create_saver(callback_name="val/lidar")
        
        elif cfg.MODEL.USE_IMAGE:
            saver_callbacks += create_saver(callback_name="val/image")
        
    trainer.train(
        dataflow=dataflow['train'],
        num_epochs=cfg.SCHEDULER.MAX_EPOCH,
        callbacks=[
            InferenceRunner(dataflow['val'], callbacks=inference_callbacks),
            InferenceRunner(dataflow['test'], callbacks=test_inference_callbacks),
            MetaInfoSaver(),
            ConsoleWriter(),
            TFEventWriterExtended(),
            JSONLWriter(),
            ProgressBar(),
            EstimatedTimeLeft()
        ] + saver_callbacks
    )

    dist.barrier()
    if dist.rank() == 0:
        wandb.finish()


def test(cfg = None, output_dir = None, run_name = "") -> None:
    os.environ[
        "TORCH_DISTRIBUTED_DEBUG"
    ] = "DETAIL"  # set to DETAIL for runtime logging
    
    dist.init()

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())
    print("world size", dist.size())
    print("local rank", dist.local_rank())
    
    if dist.rank() == 0:
        run = wandb.init(project='FusionTransformer', name=run_name, config=cfg, group=cfg["MODEL"]["TYPE"], sync_tensorboard=True)

    set_run_dir(output_dir)

    configs.update(cfg)

    seed = cfg.RNG_SEED + dist.rank() * cfg.DATALOADER.NUM_WORKERS * cfg.SCHEDULER.MAX_EPOCH
    print(seed)
    set_random_seed(seed)
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    

    dataflow = {}
    dataflow["test"] = build_dataloader(cfg, mode='test', use_distributed=True)


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
    inference_callbacks = []
    if cfg.MODEL.USE_FUSION:
        test_inference_callbacks = [SaverRestoreIOU(), MeanIoU(name='MeanIoU/test/lidar', num_classes=cfg["MODEL"]["NUM_CLASSES"], ignore_label=0, output_tensor="lidar_seg")]

    elif cfg.MODEL.USE_LIDAR:
        test_inference_callbacks = [SaverRestoreIOU(), MeanIoU(name='MeanIoU/test/lidar', num_classes=cfg["MODEL"]["NUM_CLASSES"], ignore_label=0, output_tensor="lidar_seg")]
    
    elif cfg.MODEL.USE_IMAGE:
        test_inference_callbacks = [SaverRestoreIOU(), MeanIoU(name='MeanIoU/test/lidar', num_classes=cfg["MODEL"]["NUM_CLASSES"], ignore_label=0, output_tensor="img_seg")]

    callbacks = Callbacks(test_inference_callbacks)
    callbacks._set_trainer(trainer)
    trainer.callbacks = callbacks
    trainer.dataflow = dataflow['test']

    trainer.before_train()
    trainer.before_epoch()

    model.eval()

    for feed_dict in tqdm(dataflow['test'], desc='eval'):
        output_dict = trainer.run_step(feed_dict)
        trainer.after_step(output_dict)

    trainer.after_epoch()    


    dist.barrier()
    if dist.rank() == 0:
        wandb.finish()


if __name__ == '__main__':
    main()