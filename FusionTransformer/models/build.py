from FusionTransformer.models.baseline_late import LateFusionTransformer
from FusionTransformer.models.middle_fusion import MiddleFusionTransformer
from FusionTransformer.models.early_fusion import EarlyFusionTransformer
from FusionTransformer.models.metric import SegIoU



def build_metrics(cfg):
    train_3d_metric = SegIoU(num_classes=cfg.MODEL.NUM_CLASSES, name='seg_iou_3d')
    train_2d_metric = SegIoU(num_classes=cfg.MODEL.NUM_CLASSES, name='seg_iou_2d')
    return train_2d_metric, train_3d_metric

def build_late_fusion_model(cfg):

    train_2d_metric, train_3d_metric = build_metrics(cfg)

    model = LateFusionTransformer(
        num_class=cfg.MODEL.NUM_CLASSES,
        dual_head=cfg.MODEL.DUAL_HEAD,
        backbone_2d_kwargs=cfg.MODEL,
        backbone_3d_kwargs=cfg.MODEL)

    return model, train_2d_metric, train_3d_metric


def build_middle_fusion_model(cfg):

    train_2d_metric, train_3d_metric = build_metrics(cfg)

    model = MiddleFusionTransformer(
        num_class=cfg.MODEL.NUM_CLASSES,
        dual_head=cfg.MODEL.DUAL_HEAD,
        backbone_2d_kwargs=cfg.MODEL,
        backbone_3d_kwargs=cfg.MODEL)

    return model, train_2d_metric, train_3d_metric

def build_early_fusion_model(cfg):

    train_2d_metric, train_3d_metric = build_metrics(cfg)

    model = EarlyFusionTransformer(
        num_class=cfg.MODEL.NUM_CLASSES,
        dual_head=cfg.MODEL.DUAL_HEAD,
        backbone_2d_kwargs=cfg.MODEL,
        backbone_3d_kwargs=cfg.MODEL)

    return model, train_2d_metric, train_3d_metric

def build_model(cfg):
    
    if cfg.MODEL.TYPE == "LateFusionTransformer":
        return build_late_fusion_model(cfg=cfg)
    
    if cfg.MODEL.TYPE == "MiddleFusionTransformer":
        return build_middle_fusion_model(cfg=cfg)

    if cfg.MODEL.TYPE == "EarlyFusionTransformer":
        return build_early_fusion_model(cfg)