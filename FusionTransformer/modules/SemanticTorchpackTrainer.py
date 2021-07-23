from typing import Any, Callable, Dict

import numpy as np
import torch
from torch import nn
from torchpack.train import Trainer
from torchpack.utils.typing import Optimizer, Scheduler
import torch.nn.functional as F
from FusionTransformer.data.utils.validate import map_sparse_to_org
__all__ = ['SemanticTorchpackTrainer']


class SemanticTorchpackTrainer(Trainer):

    def __init__(self, model: nn.Module, criterion: Callable,
                 optimizer: Optimizer, scheduler: Scheduler, num_workers: int,
                 seed: int, cfg: Dict) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_workers = num_workers
        self.seed = seed
        self.epoch_num = 1
        self.cfg = cfg
        if cfg.TRAIN.CLASS_WEIGHTS:
            self.class_weights = torch.tensor(cfg.TRAIN.CLASS_WEIGHTS).cuda()
        else:
            self.class_weights = None

    def _before_epoch(self) -> None:
        self.model.train()
        self.dataflow.sampler.set_epoch(self.epoch_num - 1)
        self.dataflow.worker_init_fn = lambda worker_id: np.random.seed(
            self.seed + (self.epoch_num - 1) * self.num_workers + worker_id)

    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:

        # copy data from cpu to gpu
        feed_dict['seg_label'] = feed_dict['seg_label'].cuda()

        if self.cfg.MODEL.USE_LIDAR:
            feed_dict['lidar'] = feed_dict['lidar'].cuda()

        if self.cfg.MODEL.USE_IMAGE:
            feed_dict['img'] = feed_dict['img'].cuda()

        scores = self.model(feed_dict)

        if self.model.training:
            outputs, targets = self.train_step(preds=scores, feed_dict=feed_dict)
        else:
            outputs, targets = self.eval_step(preds=scores, feed_dict=feed_dict)
        
        return {'targets': targets, **outputs}

    def train_step(self, preds, feed_dict):

        self.optimizer.zero_grad()
        # segmentation loss: cross entropy
        if self.cfg.MODEL.USE_FUSION:
            loss_3d = F.cross_entropy(preds['lidar_seg_logit'], feed_dict['seg_label'].long(), weight=self.class_weights)
            loss_2d = F.cross_entropy(preds['img_seg_logit'], feed_dict['seg_label'].long(), weight=self.class_weights)


            if self.cfg.TRAIN.FusionTransformer.lambda_xm > 0:
                # cross-modal loss: KL divergence
                seg_logit_2d = preds['img_seg_logit2'] if self.cfg.MODEL.DUAL_HEAD else preds['img_seg_logit']
                seg_logit_3d = preds['lidar_seg_logit2'] if self.cfg.MODEL.DUAL_HEAD else preds['lidar_seg_logit']

                xm_loss_2d = F.kl_div(F.log_softmax(seg_logit_2d, dim=1),
                                        F.softmax(preds['lidar_seg_logit'].detach(), dim=1),
                                        reduction='none').sum(1).mean()

                xm_loss_3d = F.kl_div(F.log_softmax(seg_logit_3d, dim=1),
                                        F.softmax(preds['img_seg_logit'].detach(), dim=1),
                                        reduction='none').sum(1).mean()

                loss_2d = ( 1 - self.cfg.TRAIN.FusionTransformer.lambda_xm ) * loss_2d + \
                          ( self.cfg.TRAIN.FusionTransformer.lambda_xm     ) * xm_loss_2d

                loss_3d = ( 1 - self.cfg.TRAIN.FusionTransformer.lambda_xm ) * loss_3d + \
                          ( self.cfg.TRAIN.FusionTransformer.lambda_xm     ) * xm_loss_3d
                          
            self.summary.add_scalar('loss_2d', loss_2d.item())
            self.summary.add_scalar('loss_3d', loss_3d.item())
            loss_2d.backward()
            loss_3d.backward()

        elif self.cfg.MODEL.USE_LIDAR:
            loss_3d = F.cross_entropy(preds['lidar_seg_logit'], feed_dict['seg_label'].long(), weight=self.class_weights)
            self.summary.add_scalar('loss_3d', loss_3d.item())
            loss_3d.backward()

        elif self.cfg.MODEL.USE_IMAGE:
            loss_2d = F.cross_entropy(preds['img_seg_logit'], feed_dict['seg_label'].long(), weight=self.class_weights)
            self.summary.add_scalar('loss_2d', loss_2d.item())
            loss_2d.backward()

        targets = feed_dict["seg_label"]
        self.optimizer.step()
        self.scheduler.step()

        return preds, targets

    def eval_step(self, preds, feed_dict):
        outputs = {}
        if self.cfg.MODEL.USE_FUSION:
            outputs['lidar_seg_logit'] = self.prepare_outputs_for_eval(feed_dict=feed_dict, preds=preds['lidar_seg_logit'].argmax(1))
            outputs['img_seg_logit'] = self.prepare_outputs_for_eval(feed_dict=feed_dict, preds=preds['img_seg_logit'].argmax(1))

        elif self.cfg.MODEL.USE_LIDAR:
            outputs['lidar_seg_logit'] = self.prepare_outputs_for_eval(feed_dict=feed_dict, preds=preds['lidar_seg_logit'].argmax(1))

        elif self.cfg.MODEL.USE_IMAGE:
            outputs['img_seg_logit'] = self.prepare_outputs_for_eval(feed_dict=feed_dict, preds=preds['img_seg_logit'].argmax(1))

        targets = self.prepare_targets_for_eval(feed_dict=feed_dict)
        return outputs, targets

    
    def prepare_targets_for_eval(self, feed_dict):
        _targets = []
        orig_seg_label = feed_dict['orig_seg_label']
        for batch_ind in range(len(orig_seg_label)):
            curr_orig_seg_label = orig_seg_label[batch_ind]
#             if self.dataflow.dataset.map_inverse_label is not None:
#                 curr_orig_seg_label = self.dataflow.dataset.map_inverse_label(curr_orig_seg_label)
            _targets.append(torch.from_numpy(curr_orig_seg_label))

        targets = torch.cat(_targets, 0).to(feed_dict["seg_label"].device)
        return targets

    def prepare_outputs_for_eval(self, feed_dict, preds):
        # get original point cloud from before voxelization
        points_idx = feed_dict['sparse_orig_points_idx']
        inverse_map = feed_dict["inverse_map"]
        _outputs = []
       
        # loop over batch
        left_idx = 0
        for batch_ind in range(len(inverse_map)):
            curr_points_idx = points_idx[batch_ind]
            # check if all points have predictions (= all voxels inside receptive field)
            assert np.all(curr_points_idx)
            curr_inverse_map = inverse_map[batch_ind]
            right_idx = left_idx + curr_points_idx.sum()
            preds_label = preds[left_idx:right_idx]
            preds_label = map_sparse_to_org(preds_label, curr_inverse_map)

#             if self.dataflow.dataset.map_inverse_label is not None:
#                 preds_label =  self.dataflow.dataset.map_inverse_label(preds_label.cpu().numpy())

            _outputs.append(preds_label)
#         _outputs = [torch.from_numpy(i) for i in _outputs]
        outputs = torch.cat(_outputs, 0)
                  
        return outputs

    def _after_epoch(self) -> None:
        self.model.eval()

    def _state_dict(self) -> Dict[str, Any]:
        state_dict = {}
        state_dict['model'] = self.model.state_dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['scheduler'] = self.scheduler.state_dict()
        return state_dict

    def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])

    def _load_previous_checkpoint(self, checkpoint_path: str) -> None:
        pass