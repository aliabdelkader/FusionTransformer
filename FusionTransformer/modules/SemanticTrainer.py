import os
import os.path as osp
import time

import logging
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from FusionTransformer.common.solver.build import build_optimizer, build_scheduler
from FusionTransformer.common.utils.checkpoint import CheckpointerV2

from FusionTransformer.common.utils.metric_logger import MetricLogger
from FusionTransformer.common.utils.torch_util import set_random_seed
from FusionTransformer.models.build import build_model
from FusionTransformer.data.build import build_dataloader
from FusionTransformer.data.utils.validate import validate
from FusionTransformer.models.losses import entropy_loss
from tqdm import tqdm
import wandb

class SemanticTrainer(object):
    def __init__(self, cfg, output_dir, run_name):
        # ---------------------------------------------------------------------------- #
        # Build models, optimizer, scheduler, checkpointer, etc.
        # ---------------------------------------------------------------------------- #
        self.cfg = cfg
        self.logger = logging.getLogger('FusionTransformer.train')
        wandb.login()
        self.run = wandb.init(project='FusionTransformer', config=self.cfg, group=self.cfg["MODEL"]["TYPE"], sync_tensorboard=True)
        set_random_seed(cfg.RNG_SEED)

        if self.cfg.MODEL.USE_IMAGE:
            self.model, self.train_2d_metric, self.train_3d_metric = build_model(cfg)
        else:
            self.model, self.train_3d_metric = build_model(cfg)

        wandb.watch(self.model)

        self.logger.info('Build model:\n{}'.format(str(self.model)))
        num_params = sum(param.numel() for param in self.model.parameters())
        print('#Parameters: {:.2e}'.format(num_params))


        self.model = self.model.cuda()

        # build optimizer
        self.optimizer = build_optimizer(cfg, self.model)

        # build lr scheduler
        self.scheduler = build_scheduler(cfg, self.optimizer)

        # build checkpointer
        # Note that checkpointer will load state_dict of model, optimizer and scheduler.
        self.checkpointer = CheckpointerV2(self.model,
                                        optimizer=self.optimizer,
                                        scheduler=self.scheduler,
                                        save_dir=output_dir,
                                        logger=self.logger,
                                        postfix='',
                                        max_to_keep=cfg.TRAIN.MAX_TO_KEEP)
        self.checkpoint_data = self.checkpointer.load(cfg.RESUME_PATH, resume=cfg.AUTO_RESUME, resume_states=cfg.RESUME_STATES)

        # build tensorboard logger (optionally by comment)
        if output_dir:
            tb_dir = osp.join(output_dir, 'tb.{:s}'.format(run_name))
            self.summary_writer = SummaryWriter(tb_dir)
        else:
            self.summary_writer = None

        start_epoch = self.checkpoint_data.get('epoch', 0)

        # build data loader
        # Reset the random seed again in case the initialization of models changes the random state.
        set_random_seed(cfg.RNG_SEED)
        self.train_dataloader = build_dataloader(cfg, mode='train') #, start_iteration=start_iteration)
        self.val_dataloader = build_dataloader(cfg, mode='val') if cfg.VAL.PERIOD > 0 else None

        self.best_metric_name = 'best_{}'.format(cfg.VAL.METRIC)
        if cfg.MODEL.USE_IMAGE == True:
            self.best_metric = {
                '2d': self.checkpoint_data.get(self.best_metric_name, None),
                '3d': self.checkpoint_data.get(self.best_metric_name, None)
            }
            self.best_metric_epoch = {'2d': -1, '3d': -1}
        else:
            self.best_metric = {
                '3d': self.checkpoint_data.get(self.best_metric_name, None)
            }
            self.best_metric_epoch = {'3d': -1}

        self.logger.info('Start training from epoch {}'.format(start_epoch))

        # logger.info('Start training from iteration {}'.format(start_iteration))

        # add metrics
        if cfg.MODEL.USE_IMAGE == True:
            self.train_metric_logger = self.init_metric_logger([self.train_2d_metric, self.train_3d_metric])
        else:
            self.train_metric_logger = self.init_metric_logger([self.train_3d_metric])

        self.val_metric_logger = MetricLogger(delimiter='  ')

        if cfg.TRAIN.CLASS_WEIGHTS:
            self.class_weights = torch.tensor(cfg.TRAIN.CLASS_WEIGHTS).cuda()
        else:
            self.class_weights = None
            
    @staticmethod
    def init_metric_logger(metric_list):
        new_metric_list = []
        for metric in metric_list:
            if isinstance(metric, (list, tuple)):
                new_metric_list.extend(metric)
            else:
                new_metric_list.append(metric)
        metric_logger = MetricLogger(delimiter='  ')
        metric_logger.add_meters(new_metric_list)
        return metric_logger

    def setup_train(self):
            # set training mode
            self.model.train()
            # reset metric
            self.train_metric_logger.reset()

    def setup_validate(self):
        # set evaluate mode
        self.model.eval()
        # reset metric
        self.val_metric_logger.reset()

    def train_step(self, data_batch):
        # copy data from cpu to gpu
        data_batch['lidar'] = data_batch['lidar'].cuda()
        data_batch['seg_label'] = data_batch['seg_label'].cuda()

        if self.cfg.MODEL.USE_IMAGE:
            data_batch['img'] = data_batch['img'].cuda()


        self.optimizer.zero_grad()

        preds = self.model(data_batch)

        # segmentation loss: cross entropy
        if self.cfg.MODEL.USE_IMAGE:
            loss_3d = F.cross_entropy(preds['lidar_seg_logit'], data_batch['seg_label'].long(), weight=self.class_weights)
            loss_2d = F.cross_entropy(preds['img_seg_logit'], data_batch['seg_label'].long(), weight=self.class_weights)
            self.train_metric_logger.update(seg_loss_2d=loss_2d.item(), seg_loss_3d=loss_3d.item())

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

                self.train_metric_logger.update(xm_loss_2d=xm_loss_2d.detach().item(),
                                        xm_loss_3d=xm_loss_3d.detach().item())
                loss_2d += self.cfg.TRAIN.FusionTransformer.lambda_xm * xm_loss_2d
                loss_3d += self.cfg.TRAIN.FusionTransformer.lambda_xm * xm_loss_3d
        else:
            loss_3d = F.cross_entropy(preds['lidar_seg_logit'], data_batch['seg_label'].long(), weight=self.class_weights)
            self.train_metric_logger.update(seg_loss_3d=loss_3d.item())

        # update metric (e.g. IoU)
        with torch.no_grad():
            self.train_3d_metric.update_dict(preds, data_batch)
            if self.cfg.MODEL.USE_IMAGE:
                self.train_2d_metric.update_dict(preds, data_batch)
            
        # backward
        if self.cfg.MODEL.USE_IMAGE:
            loss_2d.backward(retain_graph=True)
        loss_3d.backward()

        self.optimizer.step()

        if self.cfg.MODEL.USE_IMAGE:
            wandb.log({"loss_2d": loss_2d, "loss_3d": loss_3d})
        else:
            wandb.log({"loss_3d": loss_3d})
    
    def train_for_one_epoch(self, epoch):
        ###### start of training for one epoch ###########################################
        self.setup_train()
        end = time.time()
        for data_batch in tqdm(self.train_dataloader, f"training for epoch {epoch}:", total=len(self.train_dataloader)):
            self.train_step(data_batch=data_batch)
        self.scheduler.step()
        ###### end of training for one epoch ###########################################

    def update_log(self, epoch):
        if epoch == 1 or (self.cfg.TRAIN.LOG_PERIOD > 0 and epoch % self.cfg.TRAIN.LOG_PERIOD == 0):
            self.logger.info(
                self.train_metric_logger.delimiter.join(
                    [
                        'iter: {iter:4d}',
                        '{meters}',
                        'lr: {lr:.2e}',
                        'max mem: {memory:.0f}',
                    ]
                ).format(
                    iter=epoch,
                    meters=str(self.train_metric_logger),
                    lr=self.optimizer.param_groups[0]['lr'],
                    memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                )
            )

    def update_summary(self, epoch):
        if self.summary_writer is not None and self.cfg.TRAIN.SUMMARY_PERIOD > 0 and epoch % self.cfg.TRAIN.SUMMARY_PERIOD == 0:
            keywords = ('loss', 'acc', 'iou')
            for name, meter in self.train_metric_logger.meters.items():
                if all(k not in name for k in keywords):
                    continue
                self.summary_writer.add_scalar('train/' + name, meter.avg, global_step=epoch)

    def update_checkpoint(self, epoch):
        if (self.cfg.TRAIN.CHECKPOINT_PERIOD > 0 and epoch % self.cfg.TRAIN.CHECKPOINT_PERIOD == 0) or epoch == self.cfg.SCHEDULER.MAX_EPOCH:
            self.checkpoint_data['epoch'] = epoch
            if self.cfg.MODEL.USE_IMAGE:
                self.checkpoint_data["2d_" + self.best_metric_name] = self.best_metric['2d']
            self.checkpoint_data["3d_" + self.best_metric_name] = self.best_metric['3d']
            self.checkpointer.save('model{:06d}'.format(epoch), **self.checkpoint_data)

    def validate_for_one_epoch(self, epoch):
        # ---------------------------------------------------------------------------- #
        # validate for one epoch
        # ---------------------------------------------------------------------------- #
        if self.cfg.VAL.PERIOD > 0 and (epoch % self.cfg.VAL.PERIOD == 0 or epoch == self.cfg.SCHEDULER.MAX_EPOCH):
            self.setup_validate()
            validate(self.cfg, self.model, self.val_dataloader, self.val_metric_logger, self.class_weights)


    def update_validation_logging_meters(self, epoch):
        self.logger.info('Epoch[{}]-Val {}'.format(epoch, self.val_metric_logger.summary_str))

        if self.cfg.MODEL.USE_IMAGE:
            modalities = ['2d', '3d']
        else:
            modalities = ['3d']

            # best validation
        for modality in modalities:
            cur_metric_name = self.cfg.VAL.METRIC + '_' + modality
            if cur_metric_name in self.val_metric_logger.meters:
                cur_metric = self.val_metric_logger.meters[cur_metric_name].global_avg
                if self.best_metric[modality] is None or self.best_metric[modality] < cur_metric:
                    self.best_metric[modality] = cur_metric
                    self.best_metric_epoch[modality] = epoch
    
        for modality in modalities:
            self.logger.info('Best val-{}-{} = {:.2f} at epoch {}'.format(modality.upper(),
                                                                            self.cfg.VAL.METRIC,
                                                                            self.best_metric[modality] * 100,
                                                                            self.best_metric_epoch[modality]))
            wandb.log({f"Best val-{modality.upper()}-{self.cfg.VAL.METRIC}": f"{self.best_metric[modality]:.2f}"})

    def update_validation_summary(self, epoch):
        if self.summary_writer is not None:
            keywords = ('loss', 'acc', 'iou')
            for name, meter in self.val_metric_logger.meters.items():
                if all(k not in name for k in keywords):
                    continue
                self.summary_writer.add_scalar('val/' + name, meter.avg, global_step=epoch)
    
    def train(self):
        # train_iter = enumerate(train_dataloader)
            
        for epoch in tqdm(range(int(self.cfg.SCHEDULER.MAX_EPOCH)), "epoch: "):

            self.train_for_one_epoch(epoch=epoch)
            
            self.update_log(epoch=epoch)

            self.update_summary(epoch=epoch)

            self.validate_for_one_epoch(epoch=epoch)
            
            self.update_validation_logging_meters(epoch=epoch)


            self.update_validation_summary(epoch=epoch)
                                    
            # save model if best iou was in this epoch
            if  ( self.best_metric_epoch.get('2d', None) == epoch ) or ( self.best_metric_epoch.get('3d', None) == epoch ): 
                self.update_checkpoint(epoch=epoch)
                
        wandb.finish()