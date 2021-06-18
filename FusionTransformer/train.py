#!/usr/bin/env python
import os
import os.path as osp
import argparse
import logging
import time
import socket
import warnings
import torch

from FusionTransformer.modules.SemanticTrainer import SemanticTrainer
from FusionTransformer.common.utils.logger import setup_logger

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

    trainer = SemanticTrainer(cfg, output_dir, run_name)
    trainer.train()


if __name__ == '__main__':
    import sys, traceback, ipdb
    try:
        main()
    except:
        import wandb; wandb.finish()
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
