#!/usr/bin/env python
import os
import os.path as osp
import argparse
import logging
import time
import socket
import warnings

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from FusionTransformer.common.solver.build import build_optimizer, build_scheduler
from FusionTransformer.common.utils.checkpoint import CheckpointerV2
from FusionTransformer.common.utils.logger import setup_logger
from FusionTransformer.common.utils.metric_logger import MetricLogger
from FusionTransformer.common.utils.torch_util import set_random_seed
from FusionTransformer.models.build import build_model
from FusionTransformer.data.build import build_dataloader
from FusionTransformer.data.utils.validate import validate
from FusionTransformer.models.losses import entropy_loss
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='FusionTransformer training')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='',
        metavar='FILE',
        help='path to config file',
        type=str,
    )
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args


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

def setup_train(model, train_metric_logger):
        # set training mode
        model.train()
        # reset metric
        train_metric_logger.reset()

def setup_validate(model, val_metric_logger):
        # set evaluate mode
        model.eval()
        # reset metric
        val_metric_logger.reset()
def train_step(data_batch, model, optimizer, train_metric_logger, train_2d_metric, train_3d_metric, cfg, class_weights):
    # copy data from cpu to gpu
    data_batch['lidar'] = data_batch['lidar'].cuda()
    data_batch['seg_label'] = data_batch['seg_label'].cuda()
    data_batch['img'] = data_batch['img'].cuda()


    optimizer.zero_grad()

    preds = model(data_batch)

    # segmentation loss: cross entropy
    loss_2d = F.cross_entropy(preds['img_seg_logit'], data_batch['seg_label'].long(), weight=class_weights)
    loss_3d = F.cross_entropy(preds['lidar_seg_logit'], data_batch['seg_label'].long(), weight=class_weights)
    train_metric_logger.update(seg_loss_src_2d=loss_2d.item(), seg_loss_src_3d=loss_3d.item())

    if cfg.TRAIN.FusionTransformer.lambda_xm > 0:
        # cross-modal loss: KL divergence
        seg_logit_2d = preds['img_seg_logit2'] if cfg.MODEL.DUAL_HEAD else preds['img_seg_logit']
        seg_logit_3d = preds['lidar_seg_logit2'] if cfg.MODEL.DUAL_HEAD else preds['lidar_seg_logit']

        xm_loss_2d = F.kl_div(F.log_softmax(seg_logit_2d, dim=1),
                                F.softmax(preds['lidar_seg_logit'].detach(), dim=1),
                                reduction='none').sum(1).mean()

        xm_loss_3d = F.kl_div(F.log_softmax(seg_logit_3d, dim=1),
                                F.softmax(preds['img_seg_logit'].detach(), dim=1),
                                reduction='none').sum(1).mean()

        train_metric_logger.update(xm_loss_src_2d=xm_loss_2d.detach().item(),
                                xm_loss_src_3d=xm_loss_3d.detach().item())
        loss_2d += cfg.TRAIN.FusionTransformer.lambda_xm * xm_loss_2d
        loss_3d += cfg.TRAIN.FusionTransformer.lambda_xm * xm_loss_3d

    # update metric (e.g. IoU)
    with torch.no_grad():
        train_2d_metric.update_dict(preds, data_batch)
        train_3d_metric.update_dict(preds, data_batch)

    # backward
    loss_2d.backward(retain_graph=True)
    loss_3d.backward()

    optimizer.step()
    
def train_for_one_epoch(epoch, model, optimizer, scheduler, train_metric_logger, train_2d_metric, train_3d_metric, cfg, train_dataloader, class_weights):
    ###### start of training for one epoch ###########################################
    setup_train(model=model, train_metric_logger=train_metric_logger)
    end = time.time()
    for data_batch in tqdm(train_dataloader, "training for one epoch", total=len(train_dataloader)):
        train_step(data_batch=data_batch, model=model, optimizer=optimizer, train_metric_logger=train_metric_logger,
         train_2d_metric=train_2d_metric, train_3d_metric=train_3d_metric, cfg=cfg, class_weights=class_weights)
    scheduler.step()
    ###### end of training for one epoch ###########################################

def update_log(epoch, train_metric_logger, optimizer, cfg, logger):
    if epoch == 1 or (cfg.TRAIN.LOG_PERIOD > 0 and epoch % cfg.TRAIN.LOG_PERIOD == 0):
        logger.info(
            train_metric_logger.delimiter.join(
                [
                    'iter: {iter:4d}',
                    '{meters}',
                    'lr: {lr:.2e}',
                    'max mem: {memory:.0f}',
                ]
            ).format(
                iter=epoch,
                meters=str(train_metric_logger),
                lr=optimizer.param_groups[0]['lr'],
                memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
            )
        )

def update_summary(epoch, summary_writer, cfg, train_metric_logger):
    if summary_writer is not None and cfg.TRAIN.SUMMARY_PERIOD > 0 and epoch % cfg.TRAIN.SUMMARY_PERIOD == 0:
        keywords = ('loss', 'acc', 'iou')
        for name, meter in train_metric_logger.meters.items():
            if all(k not in name for k in keywords):
                continue
            summary_writer.add_scalar('train/' + name, meter.avg, global_step=epoch)

def update_checkpoint(epoch, checkpointer, checkpoint_data, best_metric, best_metric_name, cfg):
    if (cfg.TRAIN.CHECKPOINT_PERIOD > 0 and epoch % cfg.TRAIN.CHECKPOINT_PERIOD == 0) or epoch == cfg.SCHEDULER.MAX_EPOCH:
        checkpoint_data['epoch'] = epoch
        checkpoint_data["2d_" + best_metric_name] = best_metric['2d']
        checkpoint_data["3d_" + best_metric_name] = best_metric['3d']
        checkpointer.save('model{:06d}'.format(epoch), **checkpoint_data)

def validate_for_one_epoch(epoch, model, val_dataloader, val_metric_logger, cfg, class_weights):
    # ---------------------------------------------------------------------------- #
    # validate for one epoch
    # ---------------------------------------------------------------------------- #
    if cfg.VAL.PERIOD > 0 and (epoch % cfg.VAL.PERIOD == 0 or epoch == cfg.SCHEDULER.MAX_EPOCH):
        setup_validate(model=model, val_metric_logger=val_metric_logger)
        validate(cfg, model, val_dataloader, val_metric_logger, class_weights)


def update_validation_logging_meters(epoch, val_metric_logger, cfg, best_metric, logger, best_metric_epoch):
    logger.info('Epoch[{}]-Val {}'.format(
        epoch, val_metric_logger.summary_str))
        # best validation
    for modality in ['2d', '3d']:
        cur_metric_name = cfg.VAL.METRIC + '_' + modality
        if cur_metric_name in val_metric_logger.meters:
            cur_metric = val_metric_logger.meters[cur_metric_name].global_avg
            if best_metric[modality] is None or best_metric[modality] < cur_metric:
                best_metric[modality] = cur_metric
                best_metric_epoch[modality] = epoch
    

    for modality in ['2d', '3d']:
        logger.info('Best val-{}-{} = {:.2f} at epoch {}'.format(modality.upper(),
                                                                        cfg.VAL.METRIC,
                                                                        best_metric[modality] * 100,
                                                                        best_metric_epoch[modality]))


def update_validation_summary(epoch, summary_writer, val_metric_logger):
    if summary_writer is not None:
        keywords = ('loss', 'acc', 'iou')
        for name, meter in val_metric_logger.meters.items():
            if all(k not in name for k in keywords):
                continue
            summary_writer.add_scalar('val/' + name, meter.avg, global_step=epoch)


def train(cfg, output_dir='', run_name=''):
    # ---------------------------------------------------------------------------- #
    # Build models, optimizer, scheduler, checkpointer, etc.
    # ---------------------------------------------------------------------------- #
    logger = logging.getLogger('FusionTransformer.train')

    set_random_seed(cfg.RNG_SEED)

    # build 2d model
    model, train_2d_metric, train_3d_metric = build_model(cfg)

    logger.info('Build model:\n{}'.format(str(model)))
    num_params = sum(param.numel() for param in model.parameters())
    print('#Parameters: {:.2e}'.format(num_params))


    model = model.cuda()

    # build optimizer
    optimizer = build_optimizer(cfg, model)

    # build lr scheduler
    scheduler = build_scheduler(cfg, optimizer)

    # build checkpointer
    # Note that checkpointer will load state_dict of model, optimizer and scheduler.
    checkpointer = CheckpointerV2(model,
                                     optimizer=optimizer,
                                     scheduler=scheduler,
                                     save_dir=output_dir,
                                     logger=logger,
                                     postfix='',
                                     max_to_keep=cfg.TRAIN.MAX_TO_KEEP)
    checkpoint_data = checkpointer.load(cfg.RESUME_PATH, resume=cfg.AUTO_RESUME, resume_states=cfg.RESUME_STATES)

    # build tensorboard logger (optionally by comment)
    if output_dir:
        tb_dir = osp.join(output_dir, 'tb.{:s}'.format(run_name))
        summary_writer = SummaryWriter(tb_dir)
    else:
        summary_writer = None

    # ---------------------------------------------------------------------------- #
    # Train
    # ---------------------------------------------------------------------------- #

    start_epoch = checkpoint_data.get('epoch', 0)

    # build data loader
    # Reset the random seed again in case the initialization of models changes the random state.
    set_random_seed(cfg.RNG_SEED)
    train_dataloader = build_dataloader(cfg, mode='train') #, start_iteration=start_iteration)
    val_dataloader = build_dataloader(cfg, mode='val') if cfg.VAL.PERIOD > 0 else None

    best_metric_name = 'best_{}'.format(cfg.VAL.METRIC)
    best_metric = {
        '2d': checkpoint_data.get(best_metric_name, None),
        '3d': checkpoint_data.get(best_metric_name, None)
    }
    best_metric_epoch = {'2d': -1, '3d': -1}
    logger.info('Start training from epoch {}'.format(start_epoch))

    # logger.info('Start training from iteration {}'.format(start_iteration))

    # add metrics
    train_metric_logger = init_metric_logger([train_2d_metric, train_3d_metric])
    val_metric_logger = MetricLogger(delimiter='  ')

    if cfg.TRAIN.CLASS_WEIGHTS:
        class_weights = torch.tensor(cfg.TRAIN.CLASS_WEIGHTS).cuda()
    else:
        class_weights = None

    # train_iter = enumerate(train_dataloader)    
    for epoch in tqdm(range(int(cfg.SCHEDULER.MAX_EPOCH)), "epoch: "):

        train_for_one_epoch(epoch=epoch, 
                            model=model, 
                            optimizer=optimizer,
                            scheduler=scheduler,
                            train_metric_logger=train_metric_logger, 
                            train_2d_metric=train_2d_metric, 
                            train_3d_metric=train_3d_metric,
                            cfg=cfg,
                            train_dataloader=train_dataloader,
                            class_weights=class_weights)
        
        update_log(epoch=epoch, 
                   train_metric_logger=train_metric_logger, 
                   optimizer=optimizer, 
                   cfg=cfg,
                   logger=logger)

        update_summary(epoch=epoch, 
                        summary_writer=summary_writer, 
                        cfg=cfg, 
                        train_metric_logger=train_metric_logger)

        validate_for_one_epoch(epoch=epoch, 
                               model=model, 
                               val_dataloader=val_dataloader, 
                               val_metric_logger=val_metric_logger, 
                               cfg=cfg,
                               class_weights=class_weights)
        
        update_validation_logging_meters(epoch=epoch, 
                                        val_metric_logger=val_metric_logger, 
                                        cfg=cfg, 
                                        best_metric=best_metric, 
                                        logger=logger,
                                        best_metric_epoch=best_metric_epoch)


        update_validation_summary(epoch=epoch, 
                                  summary_writer=summary_writer, 
                                  val_metric_logger=val_metric_logger)
                                  
        # save model if best iou was in this epoch
        if  ( best_metric_epoch['2d'] == epoch ) or ( best_metric_epoch['3d'] == epoch ): 
            update_checkpoint(epoch=epoch, 
                            checkpointer=checkpointer, 
                            checkpoint_data=checkpoint_data,
                            best_metric=best_metric, 
                            best_metric_name=best_metric_name,
                            cfg=cfg)        


def main():
    args = parse_args()

    # load the configuration
    # import on-the-fly to avoid overwriting cfg
    from FusionTransformer.common.config import purge_cfg
    from FusionTransformer.config.FusionTransformerConfig import cfg
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    purge_cfg(cfg)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    # replace '@' with config path
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        output_dir = output_dir.replace('@', config_path.replace('configs/', ''))
        if osp.isdir(output_dir):
            warnings.warn('Output directory exists.')
        os.makedirs(output_dir, exist_ok=True)

    # run name
    timestamp = time.strftime('%m-%d_%H-%M-%S')
    hostname = socket.gethostname()
    run_name = '{:s}.{:s}'.format(timestamp, hostname)

    logger = setup_logger('FusionTransformer', output_dir, comment='train.{:s}'.format(run_name))
    logger.info('{:d} GPUs available'.format(torch.cuda.device_count()))
    logger.info(args)

    logger.info('Loaded configuration file {:s}'.format(args.config_file))
    logger.info('Running with config:\n{}'.format(cfg))

    train(cfg, output_dir, run_name)


if __name__ == '__main__':
    import sys, traceback, pdb
    try:
        main()
    except:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)