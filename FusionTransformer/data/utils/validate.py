import numpy as np
import logging
import time

import torch
import torch.nn.functional as F

from FusionTransformer.data.utils.evaluate import Evaluator
from tqdm import tqdm
def map_sparse_to_org(x, inverse_map):
    return x[inverse_map]

def validate(cfg,
             model,
             dataloader,
             val_metric_logger,
             class_weights
             ):
    logger = logging.getLogger('FusionTransformer.validate')
    logger.info('Validation')

    # evaluator
    class_names = dataloader.dataset.class_names
    class_labels = dataloader.dataset.class_labels
    evaluator_3d = Evaluator(class_names, labels=class_labels)

    if cfg.MODEL.USE_IMAGE: 
        evaluator_ensemble = Evaluator(class_names, labels=class_labels) 
        evaluator_2d = Evaluator(class_names, labels=class_labels)
    else:
        evaluator_2d = None
        evaluator_ensemble = None

    end = time.time()
    with torch.no_grad():
        for iteration, data_batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            data_time = time.time() - end
            # copy data from cpu to gpu
            if 'SCN' in cfg.DATASET.TYPE:
                if cfg.MODEL.USE_IMAGE:
                    data_batch["img"] = data_batch["img"].cuda()
                data_batch["lidar"] = data_batch["lidar"].cuda()
                data_batch['seg_label'] = data_batch['seg_label'].cuda()
            else:
                raise NotImplementedError

            # predict
            preds = model(data_batch)

            if cfg.MODEL.USE_IMAGE:
                pred_label_voxel_2d = preds['img_seg_logit'].argmax(1).cpu().numpy()
                probs_2d = F.softmax(preds['img_seg_logit'], dim=1)
                probs_3d = F.softmax(preds['lidar_seg_logit'], dim=1) 
                pred_label_voxel_ensemble = (probs_2d + probs_3d).argmax(1).cpu().numpy()

            pred_label_voxel_3d = preds['lidar_seg_logit'].argmax(1).cpu().numpy() 

            # get original point cloud from before voxelization
            seg_label = data_batch['orig_seg_label']
            points_idx = data_batch['sparse_orig_points_idx']
            inverse_map = data_batch["inverse_map"]
            # loop over batch
            left_idx = 0
            for batch_ind in range(len(seg_label)):
                curr_points_idx = points_idx[batch_ind]
                # check if all points have predictions (= all voxels inside receptive field)
                assert np.all(curr_points_idx)
                curr_inverse_map = inverse_map[batch_ind]

                curr_seg_label = seg_label[batch_ind]
                right_idx = left_idx + curr_points_idx.sum()

                pred_label_3d = pred_label_voxel_3d[left_idx:right_idx]
                pred_label_3d = map_sparse_to_org(pred_label_3d, curr_inverse_map)

                if cfg.MODEL.USE_IMAGE:
                    pred_label_2d = pred_label_voxel_2d[left_idx:right_idx]
                    pred_label_2d = map_sparse_to_org(pred_label_2d, curr_inverse_map)
                    pred_label_ensemble = pred_label_voxel_ensemble[left_idx:right_idx]
                    pred_label_ensemble = map_sparse_to_org(pred_label_ensemble, curr_inverse_map)

                if dataloader.dataset.map_inverse_label is not None:
                    curr_seg_label = dataloader.dataset.map_inverse_label(curr_seg_label)
                    pred_label_3d =  dataloader.dataset.map_inverse_label(pred_label_3d)
                    if cfg.MODEL.USE_IMAGE:
                        pred_label_2d =  dataloader.dataset.map_inverse_label(pred_label_2d)
                        pred_label_ensemble = dataloader.dataset.map_inverse_label(pred_label_ensemble)

                evaluator_3d.update(pred_label_3d, curr_seg_label)

                # evaluate
                if cfg.MODEL.USE_IMAGE:
                    evaluator_2d.update(pred_label_2d, curr_seg_label)
                    evaluator_ensemble.update(pred_label_ensemble, curr_seg_label)

                left_idx = right_idx

            seg_loss_3d = F.cross_entropy(preds['lidar_seg_logit'], data_batch['seg_label'], weight=class_weights) 
            if seg_loss_3d is not None:
                val_metric_logger.update(seg_loss_3d=seg_loss_3d)

            if cfg.MODEL.USE_IMAGE:
                seg_loss_2d = F.cross_entropy(preds['img_seg_logit'], data_batch['seg_label'], weight=class_weights)
                val_metric_logger.update(seg_loss_2d=seg_loss_2d)

            batch_time = time.time() - end
            val_metric_logger.update(time=batch_time, data=data_time)
            end = time.time()

        eval_list = []
        if evaluator_2d is not None:
            val_metric_logger.update(seg_iou_2d=evaluator_2d.overall_iou)
            eval_list.extend([('2D', evaluator_2d), ('2D+3D', evaluator_ensemble)])

        if evaluator_3d is not None:
            val_metric_logger.update(seg_iou_3d=evaluator_3d.overall_iou)
            eval_list.extend([('3D', evaluator_3d)])

        eval_list.extend([])
        for modality, evaluator in eval_list:
            logger.info('{} overall accuracy={:.2f}%'.format(modality, 100.0 * evaluator.overall_acc))
            logger.info('{} overall IOU={:.2f}'.format(modality, 100.0 * evaluator.overall_iou))
            logger.info('{} class-wise segmentation accuracy and IoU.\n{}'.format(modality, evaluator.print_table()))
