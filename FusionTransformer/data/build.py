from torch.utils.data.sampler import RandomSampler, BatchSampler
from torch.utils.data.dataloader import DataLoader, default_collate
from yacs.config import CfgNode as CN

from FusionTransformer.common.utils.torch_util import worker_init_fn
from FusionTransformer.common.utils.sampler import IterationBasedBatchSampler
from FusionTransformer.data.nuscenes.nuscenes_dataloader import NuScenesSCN
from FusionTransformer.data.semantic_kitti.semantic_kitti_dataloader import SemanticKITTISCN
from FusionTransformer.data.collate import get_collate_scn

def build_dataloader(cfg, mode='train', start_iteration=0, halve_batch_size=False):
    assert mode in ['train', 'val', 'test']
    dataset_cfg = cfg.get('DATASET')
    split = dataset_cfg[mode.upper()]
    is_train = 'train' in mode
    batch_size = cfg['TRAIN'].BATCH_SIZE if is_train else cfg['VAL'].BATCH_SIZE
    if halve_batch_size:
        batch_size = batch_size // 2

    # build dataset
    # Make a copy of dataset_kwargs so that we can pop augmentation afterwards without destroying the cfg.
    # Note that the build_dataloader fn is called twice for train and val.
    dataset_kwargs = CN(dataset_cfg.get(dataset_cfg.TYPE, dict()))
    augmentation = dataset_kwargs.pop('augmentation')
    augmentation = augmentation if is_train else dict()
    if dataset_cfg.TYPE == 'NuScenesSCN':
        dataset = NuScenesSCN(split=split,
                              output_orig=not is_train,
                              **dataset_kwargs,
                              **augmentation)
    elif dataset_cfg.TYPE == 'SemanticKITTISCN':
        dataset = SemanticKITTISCN(split=split,
                                   output_orig=not is_train,
                                   **dataset_kwargs,
                                   **augmentation)
    else:
        raise ValueError('Unsupported type of dataset: {}.'.format(dataset_cfg.TYPE))


    collate_fn = get_collate_scn(is_train=is_train)

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
