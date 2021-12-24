import torch
from torch.utils.data.dataloader import DataLoader
from yacs.config import CfgNode as CN

from FusionTransformer.common.utils.torch_util import worker_init_fn, dist_worker_init_fn
from FusionTransformer.data.nuscenes.nuscenes_dataloader import NuScenesSCN
from FusionTransformer.data.semantic_kitti.semantic_kitti_dataloader import SemanticKITTISCN
from FusionTransformer.data.semantic_kitti.debug_semantic_kitti_dataloader import DebugSemanticKITTISCN

from FusionTransformer.data.collate import get_collate_scn


def build_dataloader(cfg, mode='train', start_iteration=0, halve_batch_size=False, use_distributed=False,  seed=0, use_kfolds=False, fold=None):

    assert mode in ['train', 'val', 'test']
    dataset_cfg = cfg.get('DATASET')
    split = dataset_cfg[mode.upper()]
    is_train = 'train' in mode
    is_test = 'test' in mode

    if is_train:
        batch_size = cfg['TRAIN'].BATCH_SIZE

    elif is_test:
        batch_size = cfg['TEST'].BATCH_SIZE
    else:
        batch_size = cfg['VAL'].BATCH_SIZE

    if halve_batch_size:
        batch_size = batch_size // 2

    # build dataset
    # Make a copy of dataset_kwargs so that we can pop augmentation afterwards without destroying the cfg.
    # Note that the build_dataloader fn is called twice for train and val.
    dataset_kwargs = CN(dataset_cfg.get(dataset_cfg.TYPE, dict()))
    augmentation = dataset_kwargs.pop('augmentation')
    # augmentation = augmentation if is_train else dict()
    # if dataset_cfg.TYPE == 'NuScenesSCN':
    #     dataset = NuScenesSCN(split=split,
    #                           output_orig=not is_train,
    #                           use_kfolds=use_kfold
    #                           **dataset_kwargs,
    #                           **augmentation)
    if dataset_cfg.TYPE == 'SemanticKITTISCN':
        dataset = SemanticKITTISCN(split=split,
                                   output_orig=not is_train,
                                   use_kfolds=use_kfold,
                                   fold=fold,
                                   **dataset_kwargs,
                                   **augmentation)

    elif dataset_cfg.TYPE == 'DebugSemanticKITTISCN':
        dataset = DebugSemanticKITTISCN(split=split,
                                        output_orig=not is_train,
                                        use_kfolds=use_kfold,
                                        fold=fold,
                                        **dataset_kwargs,
                                        **augmentation)
    else:
        raise ValueError(
            'Unsupported type of dataset: {}.'.format(dataset_cfg.TYPE))

    collate_fn = get_collate_scn(is_train=is_train)

    if use_distributed:
        import torchpack.distributed as dist
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=dist.size(),
            rank=dist.rank(),
            shuffle=is_train,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            worker_init_fn=dist_worker_init_fn,
            pin_memory=True,
            collate_fn=collate_fn)

    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=False,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
            shuffle=is_train
        )

    return dataloader
