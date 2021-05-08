import os
import os.path as osp
import numpy as np
import pickle
from PIL import Image
import glob
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from FusionTransformer.data.semantic_kitti import splits
from pathlib import Path
from tqdm import tqdm
# prevent "RuntimeError: received 0 items of ancdata"
torch.multiprocessing.set_sharing_strategy('file_system')


class DummyDataset(Dataset):
    """Use torch dataloader for multiprocessing"""
    def __init__(self, root_dir, scenes, img_width, img_height):
        self.root_dir = root_dir
        self.data = []
        self.glob_frames(scenes)
        self.img_width, self.img_height = img_width, img_height

    def glob_frames(self, scenes):
        for scene in scenes:
            glob_path = osp.join(self.root_dir, 'dataset', 'sequences', scene, 'image_2', '*.png')
            cam_paths = sorted(glob.glob(glob_path))
            # load calibration
            calib = self.read_calib(osp.join(self.root_dir, 'dataset', 'sequences', scene, 'calib.txt'))
            proj_matrix = calib['P2'] @ calib['Tr']
            proj_matrix = proj_matrix.astype(np.float32)

            for cam_path in cam_paths:
                basename = osp.basename(cam_path)
                frame_id = osp.splitext(basename)[0]
                assert frame_id.isdigit()
                data = {
                    'camera_path': cam_path,
                    'lidar_path': osp.join(self.root_dir, 'dataset', 'sequences', scene, 'velodyne',
                                           frame_id + '.bin'),
                    'label_path': osp.join(self.root_dir, 'dataset', 'sequences', scene, 'labels',
                                           frame_id + '.label'),
                    'proj_matrix': proj_matrix
                }
                for k, v in data.items():
                    if isinstance(v, str):
                        if not osp.exists(v):
                            raise IOError('File not found {}'.format(v))
                self.data.append(data)

    @staticmethod
    def read_calib(calib_path):
        """
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, 'r') as f:
            for line in f.readlines():
                if line == '\n':
                    break
                key, value = line.split(':', 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        calib_out['P2'] = calib_all['P2'].reshape(3, 4)  # 3x4 projection matrix for left camera
        calib_out['Tr'] = np.identity(4)  # 4x4 matrix
        calib_out['Tr'][:3, :4] = calib_all['Tr'].reshape(3, 4)
        return calib_out

    @staticmethod
    def select_points_in_frustum(points_2d, x1, y1, x2, y2):
        """
        Select points in a 2D frustum parametrized by x1, y1, x2, y2 in image coordinates
        :param points_2d: point cloud projected into 2D
        :param points_3d: point cloud
        :param x1: left bound
        :param y1: upper bound
        :param x2: right bound
        :param y2: lower bound
        :return: points (2D and 3D) that are in the frustum
        """
        keep_ind = (points_2d[:, 0] > x1) * \
                   (points_2d[:, 1] > y1) * \
                   (points_2d[:, 0] < x2) * \
                   (points_2d[:, 1] < y2)

        return keep_ind

    def __getitem__(self, index):
        data_dict = self.data[index].copy()
        scan = np.fromfile(data_dict['lidar_path'], dtype=np.float32)
        scan = scan.reshape((-1, 4))
        points = scan[:, :3]
        label = np.fromfile(data_dict['label_path'], dtype=np.uint32)
        label = label.reshape((-1))
        label = label & 0xFFFF  # get lower half for semantics

        # load image
        image = Image.open(data_dict['camera_path'])
        image = image.crop((0, 0, self.img_width, self.img_height))
        image_size = image.size

        # project points into image
        keep_idx = points[:, 0] > 0  # only keep point in front of the vehicle
        points_hcoords = np.concatenate([points[keep_idx], np.ones([keep_idx.sum(), 1], dtype=np.float32)], axis=1) # homogeneous
        img_points = (data_dict['proj_matrix'] @ points_hcoords.T).T
        img_points = img_points[:, :2] / np.expand_dims(img_points[:, 2], axis=1)  # scale 2D points
        keep_idx_img_pts = self.select_points_in_frustum(img_points, 0, 0, *image_size)
        keep_idx[keep_idx] = keep_idx_img_pts
        # fliplr so that indexing is row, col and not col, row
        img_points = np.fliplr(img_points)


        data_dict['seg_label'] = label[keep_idx].astype(np.int16)
        data_dict['points'] = points[keep_idx]
        data_dict['feats'] = scan[keep_idx]

        data_dict['points_img'] = img_points[keep_idx_img_pts]
        data_dict['image_size'] = np.array(image_size)

        return data_dict

    def __len__(self):
        return len(self.data)


def preprocess(split_name, root_dir, out_dir, img_width, img_height):
    # pkl_data = []
    sequences = getattr(splits, split_name)

    for seq in sequences:
        dataloader = DataLoader(DummyDataset(root_dir, [seq], img_width, img_height), num_workers=5)
        num_skips = 0
        for i, data_dict in enumerate(dataloader):
            # data error leads to returning empty dict
            if not data_dict:
                print('empty dict, continue')
                num_skips += 1
                continue
            for k, v in data_dict.items():
                data_dict[k] = v[0]
            print('{}/{} {}'.format(i, len(dataloader), data_dict['lidar_path']))

            # convert to relative path
            lidar_path = data_dict['lidar_path'].replace(root_dir + '/', '')
            cam_path = data_dict['camera_path'].replace(root_dir + '/', '')

            # append data
            scan_data = {
                'points': data_dict['points'].numpy(),
                'feats':  data_dict['feats'].numpy(),
                'seg_labels': data_dict['seg_label'].numpy(),
                'points_img': data_dict['points_img'].numpy(),  # row, col format, shape: (num_points, 2)
                'lidar_path': lidar_path,
                'camera_path': cam_path,
                'image_size': tuple(data_dict['image_size'].numpy())
            }
            save_dir = osp.join(out_dir, str(seq))
            os.makedirs(save_dir, exist_ok=True)
            save_path = osp.join(save_dir, 'scan_data_{}.pkl'.format(i))
            with open(save_path, 'wb') as f:
                pickle.dump(scan_data, f)
                print('Wrote preprocessed data to ' + save_path)

    print('Skipped {} files'.format(num_skips))

def calculate_min_img_shape(root_dir):
    all_image_paths = list(Path(root_dir).rglob("dataset/sequences/**/image_2/*.png"))
    images_shapes = []
    for p in tqdm(all_image_paths):
        img = Image.open(str(p))
        W, H = img.size
        images_shapes.append((W, H))
    
    images_shapes = np.array(images_shapes)
    min_width, min_height = images_shapes.min(0)
    
    with open("image_shapes.txt", 'w') as f:
        print(" min width ", min_width, "min height: ", min_height, file=f)

    return min_width, min_height
if __name__ == '__main__':
    root_dir = '/home/user/SemanticKitti'
    out_dir = '/home/user/SemanticKitti/preprocessed'
    min_width, min_height = calculate_min_img_shape(root_dir)
    preprocess('val', root_dir, out_dir, min_width, min_height)
    preprocess('train', root_dir, out_dir, min_width, min_height)
    preprocess('test', root_dir, out_dir, min_width, min_height)
