from typing import Any, Dict

import numpy as np
import torch
import os
from torch._C import _propagate_and_assign_input_shapes
import wandb
import glob

from torchpack import distributed as dist
from torchpack.callbacks.callback import Callback
from torchpack.callbacks.writers import TFEventWriter
from torchpack.callbacks.checkpoint import SaverRestore
from torchpack.train.summary import Summary
from typing import List, Optional, Union
from torchpack.callbacks import MaxSaver
from torchpack.utils.logging import logger
from torchpack.utils import io
from pathlib import Path

__all__ = ['MeanIoU', 'iouEval', 'accEval',
           'TFEventWriterExtended', 'SummaryExtended']


class MeanIoU(Callback):
    """
    modified copy of https://github.com/mit-han-lab/spvnas/blob/master/core/callbacks.py
    """

    def __init__(self,
                 num_classes: int,
                 ignore_label: int,
                 *,
                 output_tensor: str = 'outputs',
                 target_tensor: str = 'targets',
                 name: str = 'iou') -> None:
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.name = name
        self.output_tensor = output_tensor
        self.target_tensor = target_tensor
        self.class_names = [
            "unlabeled",
            "car",
            "bicycle",
            "motorcycle",
            "truck",
            "other-vehicle",
            "person",
            "bicyclist",
            "motorcyclist",
            "road",
            "parking",
            "sidewalk",
            "other-ground",
            "building",
            "fence",
            "vegetation",
            "trunk",
            "terrain",
            "pole",
            "traffic-sign"
        ]

    def _before_epoch(self) -> None:
        self.total_seen = np.zeros(self.num_classes)
        self.total_correct = np.zeros(self.num_classes)
        self.total_positive = np.zeros(self.num_classes)

    def _after_step(self, output_dict: Dict[str, Any]) -> None:
        outputs = output_dict[self.output_tensor]
        targets = output_dict[self.target_tensor]
        outputs = outputs[targets != self.ignore_label]
        targets = targets[targets != self.ignore_label]

        for i in range(self.num_classes):
            self.total_seen[i] += torch.sum(targets == i).item()
            self.total_correct[i] += torch.sum((targets == i)
                                               & (outputs == targets)).item()
            self.total_positive[i] += torch.sum(outputs == i).item()

    def _after_epoch(self) -> None:
        for i in range(self.num_classes):
            self.total_seen[i] = dist.allreduce(self.total_seen[i],
                                                reduction='sum')
            self.total_correct[i] = dist.allreduce(self.total_correct[i],
                                                   reduction='sum')
            self.total_positive[i] = dist.allreduce(self.total_positive[i],
                                                    reduction='sum')

        ious = []

        for i in range(self.num_classes):
            if self.total_seen[i] == 0:
                ious.append(0)
            else:
                cur_iou = self.total_correct[i] / (self.total_seen[i]
                                                   + self.total_positive[i]
                                                   - self.total_correct[i])
                ious.append(cur_iou)
        miou = np.mean(ious)

        print("MeanIoU: ", ious, len(ious), miou)
        self.print_table(ious)

        if hasattr(self, 'trainer') and hasattr(self.trainer, 'summary'):
            self.trainer.summary.add_scalar(self.name, miou)
        else:
            print(ious)
            print(miou)

    def print_table(self, ious):
        from tabulate import tabulate
        header = ['Class', 'IOU']
        table = []
        table.append(["MIoU", np.mean(ious)])
        for ind, class_name in enumerate(self.class_names):
            table.append([class_name,
                          ious[ind],
                          ])
        print(tabulate(table, headers=header, tablefmt='psql', floatfmt='.3f'))


class InternalEval(Callback):
    """
    modified copy of https://github.com/PRBonn/lidar-bonnetal/blob/5a5f4b180117b08879ec97a3a05a3838bce6bb0f/train/tasks/semantic/modules/ioueval.py
    """

    def __init__(self,
                 output_tensor: str = 'outputs',
                 target_tensor: str = 'targets',
                 name: str = 'iou',
                 n_classes: int = 1,
                 device='cpu',
                 ignore=None):
        self.name = name
        self.output_tensor = output_tensor
        self.target_tensor = target_tensor
        self.n_classes = n_classes
        self.device = device
        # if ignore is larger than n_classes, consider no ignoreIndex
        self.ignore = torch.tensor(ignore).long()
        self.include = torch.tensor(
            [n for n in range(self.n_classes) if n not in self.ignore]).long()
        print("[IOU EVAL] IGNORE: ", self.ignore)
        print("[IOU EVAL] INCLUDE: ", self.include)
        self.reset()

    def num_classes(self):
        return self.n_classes

    def reset(self):
        self.conf_matrix = torch.zeros(
            (self.n_classes, self.n_classes), device=self.device).long()
        self.ones = None
        self.last_scan_size = None  # for when variable scan size is used

    def _before_epoch(self):
        return self.reset()

    def _after_step(self, output_dict: Dict[str, Any]):
        outputs = output_dict[self.output_tensor]
        targets = output_dict[self.target_tensor]
        return self.addBatch(x=outputs, y=targets)

    def addBatch(self, x, y):  # x=preds, y=targets
        # if numpy, pass to pytorch
        # to tensor
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(np.array(x)).long().to(self.device)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(np.array(y)).long().to(self.device)

        # sizes should be "batch_size x H x W"
        x_row = x.reshape(-1)  # de-batchify
        y_row = y.reshape(-1)  # de-batchify

        # idxs are labels and predictions
        idxs = torch.stack([x_row, y_row], dim=0).long()

        # ones is what I want to add to conf when I
        if self.ones is None or self.last_scan_size != idxs.shape[-1]:
            self.ones = torch.ones((idxs.shape[-1]), device=self.device).long()
            self.last_scan_size = idxs.shape[-1]

        # make confusion matrix (cols = gt, rows = pred)
        self.conf_matrix = self.conf_matrix.index_put_(
            tuple(idxs), self.ones, accumulate=True)

        # print(self.tp.shape)
        # print(self.fp.shape)
        # print(self.fn.shape)

    def getStats(self):
        conf = dist.allreduce(self.conf_matrix,
                              reduction='sum')
        # remove fp and fn from confusion on the ignore classes cols and rows
#         conf = self.conf_matrix.clone().double()

        conf[self.ignore] = 0
        conf[:, self.ignore] = 0

        # get the clean stats
        tp = conf.diag()
        fp = conf.sum(dim=1) - tp
        fn = conf.sum(dim=0) - tp
        return tp, fp, fn

    def getIoU(self):
        tp, fp, fn = self.getStats()
        intersection = tp
        union = tp + fp + fn + 1e-15
        iou = intersection / union
        print("iouEval: ", iou, iou.shape)
        iou_mean = (intersection[self.include] / union[self.include]).mean()
        return iou_mean, iou  # returns "iou mean", "iou per class" ALL CLASSES

    def getacc(self):
        tp, fp, fn = self.getStats()
        total_tp = tp.sum()
        total = tp[self.include].sum() + fp[self.include].sum() + 1e-15
        acc_mean = total_tp / total
        return acc_mean  # returns "acc mean"


class iouEval(InternalEval):
    def _after_epoch(self):
        iou_mean, iou = self.getIoU()
        if hasattr(self, 'trainer') and hasattr(self.trainer, 'summary'):
            self.trainer.summary.add_scalar(self.name, iou_mean.item())
        else:
            print("mean iou", iou_mean.item())
            print("iou", iou.item())


class accEval(InternalEval):
    def _after_epoch(self):
        acc_mean = self.getacc()
        if hasattr(self, 'trainer') and hasattr(self.trainer, 'summary'):
            self.trainer.summary.add_scalar(self.name, acc_mean.item())
        else:
            print("mean acc", acc_mean.item())


class WandbMaxSaver(MaxSaver):
    def _trigger(self):
        if self.scalar not in self.trainer.summary:
            logger.warning(
                f'`{self.scalar}` has not been added to `trainer.summary`.')
            return
        step, value = self.trainer.summary[self.scalar][-1]

        if self.step is not None and step <= self.step:
            logger.warning(
                f'`{self.scalar}` has not been updated since last trigger.')
            return
        self.step = step

        if self.best is None or (self.extreme == 'min' and value < self.best[1]) \
                or (self.extreme == 'max' and value > self.best[1]):
            self.best = (step, value)
            save_path = os.path.join(self.save_dir, self.name + '.pt')
            try:
                io.save(save_path, self.trainer.state_dict())
                wandb.save(save_path)
            except OSError:
                logger.exception(
                    f'Error occurred when saving checkpoint "{save_path}".')
            else:
                logger.info(f'Checkpoint saved: "{save_path}" ({value:.5g}).')

        if self.best is not None:
            self.trainer.summary.add_scalar(self.scalar + '/' + self.extreme,
                                            self.best[1])


class TFEventWriterExtended(TFEventWriter):
    """
    Write summaries to TensorFlow event file per epoch
    """
    WANDB_MAX_HIST_BIN = 512

    def _add_scalar(self, name: str, scalar: Union[int, float]) -> None:
        self.writer.add_scalar(name, scalar, self.trainer.epoch_num)

    def _add_image(self, name: str, tensor: np.ndarray) -> None:
        self.writer.add_image(name, tensor, self.trainer.epoch_num)

    def add_weights_histogram(self) -> None:
        if self.enabled:
            for name, weight in self.trainer.model.named_parameters():
                if weight is not None:
                    self.writer.add_histogram(
                        f"{name}/weight", weight, self.trainer.global_step, max_bins=self.WANDB_MAX_HIST_BIN)

    def add_grads_histogram(self) -> None:
        if self.enabled:
            for name, weight in self.trainer.model.named_parameters():
                if weight.grad is not None:
                    self.writer.add_histogram(
                        f'{name}/grad', weight.grad, self.trainer.global_step, max_bins=self.WANDB_MAX_HIST_BIN)

    def _after_train(self) -> None:
        self.writer.close()


class SummaryExtended(Summary):
    def add_weights_histogram(self):
        for writer in self.writers:
            if isinstance(writer, TFEventWriterExtended):
                writer.add_weights_histogram()

    def add_grads_histogram(self):
        for writer in self.writers:
            if isinstance(writer, TFEventWriterExtended):
                writer.add_grads_histogram()


class SaverRestoreIOU(SaverRestore):
    def _before_train(self) -> None:
        checkpoints = glob.glob(os.path.join(
            self.load_dir, 'max-MeanIoU-*.pt'))
        if not checkpoints:
            logger.warning(f'No checkpoints found: "{self.load_dir}".')
            return

        load_path = max(checkpoints, key=os.path.getmtime)
        try:
            state_dict = io.load(load_path, map_location='cpu')
            self.trainer.load_state_dict(state_dict)
        except OSError:
            logger.exception(
                f'Error occurred when loading checkpoint "{load_path}".')
        else:
            logger.info(f'Checkpoint loaded: "{load_path}".')



class SavePredictions(Callback):
    """
    """

    def __init__(self,
                 ignore_label: int,
                 *,
                 output_tensor: str = 'outputs',
                 target_tensor: str = 'targets',
                 output_path: str = "predictions_dir",
                 save_targets: bool = False,
                 save_targets_path: str = "",
                 name: str = 'predictions') -> None:
        self.ignore_label = ignore_label
        self.name = name
        self.output_tensor = output_tensor
        self.target_tensor = target_tensor
        self.output_path = output_path
        self.save_targets = save_targets
        self.save_targets_path = save_targets_path


    def _after_step(self, output_dict: Dict[str, Any]) -> None:
        outputs = output_dict[self.output_tensor]
        targets = output_dict[self.target_tensor]
        outputs = outputs[targets != self.ignore_label]
        targets = targets[targets != self.ignore_label]

        assert len(output_dict["seq"]) == 1, f"SavePredictions assumes batch size = 1, expected len(seq) = 1, obtained  {len(output_dict['seq'])}"
        assert len(output_dict["filename"]) == 1, f"SavePredictions assumes batch size = 1, expected len(filenames) = 1, obtained  {len(output_dict['filename'])}"

        seq, filename  = output_dict["seq"][0], output_dict["filename"][0]
        path = Path(self.output_path) / seq / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        outputs_cpu = outputs.clone().detach().cpu().numpy()
        np.save(str(path), outputs_cpu)

        if self.save_targets:
            path = Path(self.save_targets_path) / seq / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            targets_cpu = targets.clone().detach().cpu().numpy()
            np.save(str(path), targets_cpu)