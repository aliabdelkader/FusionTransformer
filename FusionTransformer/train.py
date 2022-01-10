#!/usr/bin/env python
import os
import os.path as osp
import argparse
import logging
import time
import warnings
import torch

from FusionTransformer.modules import TorchpackInterface
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
    parser.add_argument(
        '--use_torchpack',
        help='use torchpack for training',
        default=False,
    )
    parser.add_argument(
        '--use_torchpack_test',
        help='use torchpack for testing',
        default=False,
    )
    parser.add_argument(
        '--run_name',
        help='set name for the run',
        default=None,
    )
    parser.add_argument(
        '--fold',
        help='fold to run',
        default=None,
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

    if args.run_name is None:
        # run name
        timestamp = time.strftime('MONTH_%m_DAY_%d_HOUR_%H_MIN_%M_SEC_%S')
        run_name = '{:s}'.format(timestamp)
    else:
        run_name = args.run_name

    output_dir = os.path.join(output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    print("output dir",  output_dir)

    # logger = setup_logger('FusionTransformer', output_dir, comment='{}.train.{:s}'.format(cfg["MODEL"]["TYPE"], run_name))
    # logger.info('{:d} GPUs available'.format(torch.cuda.device_count()))
    # logger.info(args)

    # logger.info('Loaded configuration file {:s}'.format(args.config_file))
    # logger.info('Running with config:\n{}'.format(cfg))

    if args.use_torchpack:
        TorchpackInterface.main(cfg=cfg, output_dir=output_dir, run_name=run_name)
    elif args.use_torchpack_test:
        TorchpackInterface.test(cfg=cfg, output_dir=output_dir, run_name=run_name, fold=args.fold)

    else:
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
