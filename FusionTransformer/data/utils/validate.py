import numpy as np
import logging
import time

import torch
import torch.nn.functional as F

from FusionTransformer.data.utils.evaluate import Evaluator

def map_sparse_to_org(x, inverse_map):
    return x[inverse_map]

def validate(cfg,
             model,
             dataloader,
             val_metric_logger,
             class_weights
            #  pselab_path=None
             ):
    logger = logging.getLogger('FusionTransformer.validate')
    logger.info('Validation')

    # evaluator
    class_names = dataloader.dataset.class_names
    class_labels = dataloader.dataset.class_labels
    evaluator_2d = Evaluator(class_names, labels=class_labels)
    evaluator_3d = Evaluator(class_names, labels=class_labels) #if model_3d else None
    evaluator_ensemble = Evaluator(class_names, labels=class_labels) # if model_3d else None

    pselab_data_list = []

    end = time.time()
    with torch.no_grad():
        for iteration, data_batch in enumerate(dataloader):
            data_time = time.time() - end
            # copy data from cpu to gpu
            if 'SCN' in cfg.DATASET.TYPE:
                data_batch["img"] = data_batch["img"].cuda()
                data_batch["lidar"] = data_batch["lidar"].cuda()
                # data_batch['x'][1] = data_batch['x'][1].cuda()
                data_batch['seg_label'] = data_batch['seg_label'].cuda()
                # data_batch['img'] = data_batch['img'].cuda()
            else:
                raise NotImplementedError

            # predict
            preds = model(data_batch)

            # preds_2d = model_2d(data_batch)
            # preds_3d = model_3d(data_batch) if model_3d else None

            pred_label_voxel_2d = preds['img_seg_logit'].argmax(1).cpu().numpy()
            pred_label_voxel_3d = preds['lidar_seg_logit'].argmax(1).cpu().numpy() 

            # softmax average (ensembling)
            probs_2d = F.softmax(preds['img_seg_logit'], dim=1)
            probs_3d = F.softmax(preds['lidar_seg_logit'], dim=1) 
            pred_label_voxel_ensemble = (probs_2d + probs_3d).argmax(1).cpu().numpy() 


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
                pred_label_2d = pred_label_voxel_2d[left_idx:right_idx]
                pred_label_2d = map_sparse_to_org(pred_label_2d, curr_inverse_map)

                pred_label_3d = pred_label_voxel_3d[left_idx:right_idx] #if model_3d else None
                pred_label_3d = map_sparse_to_org(pred_label_3d, curr_inverse_map)

                pred_label_ensemble = pred_label_voxel_ensemble[left_idx:right_idx]# if model_3d else None
                pred_label_ensemble = map_sparse_to_org(pred_label_ensemble, curr_inverse_map)

                if dataloader.dataset.map_inverse_label is not None:
                    curr_seg_label = dataloader.dataset.map_inverse_label(curr_seg_label)
                    pred_label_2d =  dataloader.dataset.map_inverse_label(pred_label_2d)
                    pred_label_3d =  dataloader.dataset.map_inverse_label(pred_label_3d)
                    pred_label_ensemble = dataloader.dataset.map_inverse_label(pred_label_ensemble)
                # evaluate
                evaluator_2d.update(pred_label_2d, curr_seg_label)
                # if model_3d:
                evaluator_3d.update(pred_label_3d, curr_seg_label)
                evaluator_ensemble.update(pred_label_ensemble, curr_seg_label)

                # if pselab_path is not None:
                #     assert np.all(pred_label_2d >= 0)
                #     curr_probs_2d = probs_2d[left_idx:right_idx]
                #     curr_probs_3d = probs_3d[left_idx:right_idx] if model_3d else None
                #     pselab_data_list.append({
                #         'probs_2d': curr_probs_2d[range(len(pred_label_2d)), pred_label_2d].cpu().numpy(),
                #         'pseudo_label_2d': pred_label_2d.astype(np.uint8),
                #         'probs_3d': curr_probs_3d[range(len(pred_label_3d)), pred_label_3d].cpu().numpy() if model_3d else None,
                #         'pseudo_label_3d': pred_label_3d.astype(np.uint8) if model_3d else None
                #     })

                left_idx = right_idx

            seg_loss_2d = F.cross_entropy(preds['img_seg_logit'], data_batch['seg_label'], weight=class_weights)
            seg_loss_3d = F.cross_entropy(preds['lidar_seg_logit'], data_batch['seg_label'], weight=class_weights) # if model_3d else None
            val_metric_logger.update(seg_loss_2d=seg_loss_2d)
            if seg_loss_3d is not None:
                val_metric_logger.update(seg_loss_3d=seg_loss_3d)

            batch_time = time.time() - end
            val_metric_logger.update(time=batch_time, data=data_time)
            end = time.time()

            # log
            cur_iter = iteration + 1
            if cur_iter == 1 or (cfg.VAL.LOG_PERIOD > 0 and cur_iter % cfg.VAL.LOG_PERIOD == 0):
                logger.info(
                    val_metric_logger.delimiter.join(
                        [
                            'iter: {iter}/{total_iter}',
                            '{meters}',
                            'max mem: {memory:.0f}',
                        ]
                    ).format(
                        iter=cur_iter,
                        total_iter=len(dataloader),
                        meters=str(val_metric_logger),
                        memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                    )
                )

        val_metric_logger.update(seg_iou_2d=evaluator_2d.overall_iou)
        if evaluator_3d is not None:
            val_metric_logger.update(seg_iou_3d=evaluator_3d.overall_iou)

        eval_list = [('2D', evaluator_2d),('3D', evaluator_3d), ('2D+3D', evaluator_ensemble)]
        # if model_3d:
        eval_list.extend([])
        for modality, evaluator in eval_list:
            logger.info('{} overall accuracy={:.2f}%'.format(modality, 100.0 * evaluator.overall_acc))
            logger.info('{} overall IOU={:.2f}'.format(modality, 100.0 * evaluator.overall_iou))
            logger.info('{} class-wise segmentation accuracy and IoU.\n{}'.format(modality, evaluator.print_table()))

        # if pselab_path is not None:
        #     np.save(pselab_path, pselab_data_list)
        #     logger.info('Saved pseudo label data to {}'.format(pselab_path))
