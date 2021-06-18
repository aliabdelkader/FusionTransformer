"""FusionTransformer experiments configuration"""
import os.path as osp

from FusionTransformer.common.config.base import CN, _C

# public alias
cfg = _C
_C.VAL.METRIC = 'seg_iou'

# ---------------------------------------------------------------------------- #
# Specific train options
# ---------------------------------------------------------------------------- #
_C.TRAIN.CLASS_WEIGHTS = []

# ---------------------------------------------------------------------------- #
# FusionTransformer options
# ---------------------------------------------------------------------------- #
_C.TRAIN.FusionTransformer = CN()
_C.TRAIN.FusionTransformer.lambda_xm = 0.0

# ---------------------------------------------------------------------------- #
# Datasets
# ---------------------------------------------------------------------------- #
_C.DATASET = CN()
_C.DATASET.TYPE = ''
_C.DATASET.TRAIN = tuple()
_C.DATASET.TRAIN = tuple()
_C.DATASET.VAL = tuple()
_C.DATASET.TEST = tuple()

# NuScenesSCN
_C.DATASET.NuScenesSCN = CN()
_C.DATASET.NuScenesSCN.preprocess_dir = ''
_C.DATASET.NuScenesSCN.nuscenes_dir = ''
_C.DATASET.NuScenesSCN.merge_classes = True
# 3D
_C.DATASET.NuScenesSCN.scale = 20
_C.DATASET.NuScenesSCN.full_scale = 4096
# 2D
_C.DATASET.NuScenesSCN.use_image = True
_C.DATASET.NuScenesSCN.resize = (224, 224)
_C.DATASET.NuScenesSCN.image_normalizer = ()
# 3D augmentation
_C.DATASET.NuScenesSCN.augmentation = CN()
_C.DATASET.NuScenesSCN.augmentation.noisy_rot = 0.1
_C.DATASET.NuScenesSCN.augmentation.flip_x = 0.5
_C.DATASET.NuScenesSCN.augmentation.rot_z = 6.2831  # 2 * pi
_C.DATASET.NuScenesSCN.augmentation.transl = True
# 2D augmentation
_C.DATASET.NuScenesSCN.augmentation.fliplr = 0.5
_C.DATASET.NuScenesSCN.augmentation.color_jitter = (0.4, 0.4, 0.4)

# SemanticKITTISCN
_C.DATASET.SemanticKITTISCN = CN()
_C.DATASET.SemanticKITTISCN.preprocess_dir = ''
_C.DATASET.SemanticKITTISCN.semantic_kitti_dir = ''
# _C.DATASET.SemanticKITTISCN.merge_classes = True
# 3D
_C.DATASET.SemanticKITTISCN.scale = 20
_C.DATASET.SemanticKITTISCN.full_scale = 4096
# 2D
_C.DATASET.SemanticKITTISCN.image_normalizer = ()
_C.DATASET.SemanticKITTISCN.image_width = 1226
_C.DATASET.SemanticKITTISCN.image_height = 370
# 3D augmentation
_C.DATASET.SemanticKITTISCN.augmentation = CN()
_C.DATASET.SemanticKITTISCN.augmentation.noisy_rot = 0.1
_C.DATASET.SemanticKITTISCN.augmentation.flip_y = 0.5
_C.DATASET.SemanticKITTISCN.augmentation.rot_z = 6.2831  # 2 * pi
_C.DATASET.SemanticKITTISCN.augmentation.transl = True
# 2D augmentation
_C.DATASET.SemanticKITTISCN.augmentation.bottom_crop = (480, 302)
_C.DATASET.SemanticKITTISCN.augmentation.fliplr = 0.5
_C.DATASET.SemanticKITTISCN.augmentation.color_jitter = (0.4, 0.4, 0.4)

# ---------------------------------------------------------------------------- #
# Model 3D
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.TYPE = ''
_C.MODEL.CKPT_PATH = ''
_C.MODEL.NUM_CLASSES = 20
_C.MODEL.DUAL_HEAD = True
_C.MODEL.USE_IMAGE = True

# ----------------------------------------------------------------------------- #
# SCN options
# ----------------------------------------------------------------------------- #
_C.MODEL.block_number = [11]
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# @ will be replaced by config path
_C.OUTPUT_DIR = osp.expanduser('../logs/FusionTransformer/@')
