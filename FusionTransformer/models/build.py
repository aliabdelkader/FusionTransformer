from FusionTransformer.models.baseline_late import LateFusionTransformer
from FusionTransformer.models.metric import SegIoU


def build_model_2d(cfg):
    model = Net2DSeg(num_classes=cfg.MODEL_2D.NUM_CLASSES,
                     backbone_2d=cfg.MODEL_2D.TYPE,
                     backbone_2d_kwargs=cfg.MODEL_2D[cfg.MODEL_2D.TYPE],
                     dual_head=cfg.MODEL_2D.DUAL_HEAD
                     )
    train_metric = SegIoU(cfg.MODEL_2D.NUM_CLASSES, name='seg_iou_2d')
    return model, train_metric


def build_model_3d(cfg):
    model = Net3DSeg(num_classes=cfg.MODEL_3D.NUM_CLASSES,
                     backbone_3d=cfg.MODEL_3D.TYPE,
                     backbone_3d_kwargs=cfg.MODEL_3D[cfg.MODEL_3D.TYPE],
                     dual_head=cfg.MODEL_3D.DUAL_HEAD
                     )
    train_metric = SegIoU(cfg.MODEL_3D.NUM_CLASSES, name='seg_iou_3d')
    return model, train_metric

def build_late_fusion_model(cfg):
    model = LateFusionTransformer(
        num_class=cfg.MODEL.NUM_CLASSES,
        dual_head=cfg.MODEL.DUAL_HEAD,
        backbone_2d_kwargs=cfg.MODEL,
        backbone_3d_kwargs=cfg.MODEL)
    
    train_3d_metric = SegIoU(cfg.MODEL.NUM_CLASSES, name='seg_iou_3d')
    train_2d_metric = SegIoU(cfg.MODEL.NUM_CLASSES, name='seg_iou_2d')
    return model, train_2d_metric, train_3d_metric


def build_model(cfg):
    if cfg.MODEL.TYPE == "LateFusionTransformer":
        return build_late_fusion_model(cfg=cfg)
