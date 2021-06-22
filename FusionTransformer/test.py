#!/usr/bin/env python
import os
import os.path as osp
import argparse
import logging
import time
import socket
import warnings

import torch

from FusionTransformer.common.utils.checkpoint import CheckpointerV2
from FusionTransformer.common.utils.logger import setup_logger
from FusionTransformer.common.utils.metric_logger import MetricLogger
from FusionTransformer.common.utils.torch_util import set_random_seed
from FusionTransformer.models.build import build_model
from FusionTransformer.data.build import build_dataloader
from FusionTransformer.data.utils.validate import validate


def parse_args():
    parser = argparse.ArgumentParser(description='FusionTransformer test')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='',
        metavar='FILE',
        help='path to config file',
        type=str,
    )
    parser.add_argument('--ckpt', type=str, help='path to checkpoint file of the model')
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args


def test(cfg, args, output_dir=''):
    logger = logging.getLogger('FusionTransformer.{}.test'.format(self.cfg["MODEL"]["TYPE"]))

    # build 2d model
    model = build_model(cfg)[0]

    model = model.cuda()

    # build checkpointer
    checkpointer = CheckpointerV2(model, save_dir=output_dir, logger=logger)

    if args.ckpt:
        # load weight if specified
        weight_path = args.ckpt.replace('@', output_dir)
        checkpointer.load(weight_path, resume=False)
    else:
        # load last checkpoint
        checkpointer.load(None, resume=True)

    # build dataset
    test_dataloader = build_dataloader(cfg, mode='test')

    # ---------------------------------------------------------------------------- #
    # Test
    # ---------------------------------------------------------------------------- #

    set_random_seed(cfg.RNG_SEED)
    test_metric_logger = MetricLogger(delimiter='  ')
    model.eval()

    if cfg.TRAIN.CLASS_WEIGHTS:
        class_weights = torch.tensor(cfg.TRAIN.CLASS_WEIGHTS).cuda()
    else:
        class_weights = None


    validate(cfg=cfg, model=model, dataloader=test_dataloader, val_metric_logger=test_metric_logger, class_weights=class_weights)


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
        if not osp.isdir(output_dir):
            warnings.warn('Make a new directory: {}'.format(output_dir))
            os.makedirs(output_dir)

    # run name
    timestamp = time.strftime('%m-%d_%H-%M-%S')
    hostname = socket.gethostname()
    run_name = '{:s}.{:s}'.format(timestamp, hostname)

    logger = setup_logger('FusionTransformer', output_dir, comment='{}.test.{:s}'.format(cfg["MODEL"]["TYPE"], run_name))
    logger.info('{:d} GPUs available'.format(torch.cuda.device_count()))
    logger.info(args)

    logger.info('Loaded configuration file {:s}'.format(args.config_file))
    logger.info('Running with config:\n{}'.format(cfg))

    test(cfg, args, output_dir)


if __name__ == '__main__':
    main()
