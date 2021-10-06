import os.path as osp
import pickle
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchsparse.sparse_tensor import SparseTensor
from torchsparse.point_tensor import PointTensor

from FusionTransformer.data.utils.augmentation_3d import augment_and_scale_3d
from torchsparse.utils import sparse_quantize
import yaml
from os.path import dirname, realpath
from pathlib import Path
from FusionTransformer.data.semantic_kitti import splits
from FusionTransformer.data.semantic_kitti.semantic_kitti_dataloader import SemanticKITTIBase, SemanticKITTISCN
# class SemanticKITTIBase(Dataset):
#     """SemanticKITTI dataset"""

#     # # https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti.yaml
#     # id_to_class_name = {
#     #     0: "unlabeled",
#     #     1: "outlier",
#     #     10: "car",
#     #     11: "bicycle",
#     #     13: "bus",
#     #     15: "motorcycle",
#     #     16: "on-rails",
#     #     18: "truck",
#     #     20: "other-vehicle",
#     #     30: "person",
#     #     31: "bicyclist",
#     #     32: "motorcyclist",
#     #     40: "road",
#     #     44: "parking",
#     #     48: "sidewalk",
#     #     49: "other-ground",
#     #     50: "building",
#     #     51: "fence",
#     #     52: "other-structure",
#     #     60: "lane-marking",
#     #     70: "vegetation",
#     #     71: "trunk",
#     #     72: "terrain",
#     #     80: "pole",
#     #     81: "traffic-sign",
#     #     99: "other-object",
#     #     252: "moving-car",
#     #     253: "moving-bicyclist",
#     #     254: "moving-person",
#     #     255: "moving-motorcyclist",
#     #     256: "moving-on-rails",
#     #     257: "moving-bus",
#     #     258: "moving-truck",
#     #     259: "moving-other-vehicle",
#     # }

#     # class_name_to_id = {v: k for k, v in id_to_class_name.items()}

#     def __init__(self,
#                  split,
#                  preprocess_dir,
#                  debug=False
#                  ):

#         self.split = split
#         self.preprocess_dir = preprocess_dir

#         print("Initialize SemanticKITTI dataloader")

#         assert isinstance(split, tuple)
#         print('Load', split)
        
#         self.data_paths = []
#         if debug:
#             split_sequences = getattr(splits.debug, split[0])
#         else:
#             split_sequences = getattr(splits.regular, split[0])

#         for seq in split_sequences:
#             split_path = Path(self.preprocess_dir) / seq
#             self.data_paths.extend(sorted(list(split_path.rglob("*.pkl"))))
#         # for curr_split in split:
#         #     with open(osp.join(self.preprocess_dir, curr_split + '.pkl'), 'rb') as f:
#         #         self.data.extend(pickle.load(f))

#         self.semantic_kitti_config_dict = yaml.safe_load(open(dirname(realpath(__file__)) + "/semantic_kitti_label.yaml", 'r'))
#         self.class_names = [self.semantic_kitti_config_dict["labels"][k] for k in self.semantic_kitti_config_dict["learning_map_inv"].values()]
#         self.class_labels = list (self.semantic_kitti_config_dict["learning_map_inv"].copy().values())
#         self.map_label = np.vectorize(lambda org_label: self.semantic_kitti_config_dict["learning_map"][org_label])
#         self.map_inverse_label = np.vectorize(lambda learning_label: self.semantic_kitti_config_dict["learning_map_inv"][learning_label])

#     def __getitem__(self, index):
#         raise NotImplementedError

#     def __len__(self):
#         return len(self.data_paths)


class DebugSemanticKITTISCN(SemanticKITTIBase):
    def __init__(self,
                 split,
                 preprocess_dir,
                 semantic_kitti_dir='',
                #  pselab_paths=None,
                #  merge_classes=False,
                 scale=20,
                 full_scale=4096,
                 image_normalizer=None,
                 noisy_rot=0.0,  # 3D augmentation
                 flip_y=0.0,  # 3D augmentation
                 rot_z=0.0,  # 3D augmentation
                 transl=False,  # 3D augmentation
                 bottom_crop=tuple(),  # 2D augmentation (also effects 3D)
                 fliplr=0.0,  # 2D augmentation
                 color_jitter=None,  # 2D augmentation
                 output_orig=False,
                 image_width=1226,
                 image_height=370,
                 debug=False
                 ):
        super().__init__(split,
                         preprocess_dir,
                         debug=debug
                         )


        self.semantic_kitti_dir = semantic_kitti_dir
        self.output_orig = output_orig

        # point cloud parameters
        self.scale = scale
        self.full_scale = full_scale
        # 3D augmentation
        self.noisy_rot = noisy_rot
        self.flip_y = flip_y
        self.rot_z = rot_z
        self.transl = transl

        # image parameters
        self.image_normalizer = image_normalizer
        # 2D augmentation
        self.bottom_crop = bottom_crop
        self.fliplr = fliplr
        self.color_jitter = T.ColorJitter(*color_jitter) if color_jitter else None
        self.image_width = image_width
        self.image_height = image_height
        self.data_paths = self.data_paths[:2]
        
    def __getitem__(self, index):
        data_path = str(self.data_paths[index])
        with open(data_path, 'rb') as data_file:
            data_dict = pickle.load(data_file)

        points = data_dict['points'].copy()
        feats = data_dict['feats'].copy()
        seg_label = data_dict['seg_labels'].astype(np.int64)

        if self.map_label is not None:
            seg_label = self.map_label(seg_label)

        out_dict = {}

        keep_idx = np.ones(len(points), dtype=np.bool)
        points_img = data_dict['points_img'].copy()
        img_path = osp.join(self.semantic_kitti_dir, data_dict['camera_path'])
        image = Image.open(img_path).crop((0, 0, self.image_width, self.image_height))

        if self.bottom_crop is not None:
            # self.bottom_crop is a tuple (crop_width, crop_height)
            left = int(np.random.rand() * (image.size[0] + 1 - self.bottom_crop[0]))
            right = left + self.bottom_crop[0]
            top = image.size[1] - self.bottom_crop[1]
            bottom = image.size[1]

            # update image points
            keep_idx = points_img[:, 0] >= top
            keep_idx = np.logical_and(keep_idx, points_img[:, 0] < bottom)
            keep_idx = np.logical_and(keep_idx, points_img[:, 1] >= left)
            keep_idx = np.logical_and(keep_idx, points_img[:, 1] < right)

            # crop image
            image = image.crop((left, top, right, bottom))
            points_img = points_img[keep_idx]
            points_img[:, 0] -= top
            points_img[:, 1] -= left

            # update point cloud
            points = points[keep_idx]
            seg_label = seg_label[keep_idx]
            feats = feats[keep_idx]

        img_indices = points_img.astype(np.int64)

        # 2D augmentation
        if self.color_jitter is not None:
            image = self.color_jitter(image)
        # PIL to numpy
        image = np.array(image, dtype=np.float32, copy=False)
        # 2D augmentation
        if (self.fliplr is not None) and (np.random.rand() < self.fliplr):
            image = np.ascontiguousarray(np.fliplr(image))
            img_indices[:, 1] = image.shape[1] - 1 - img_indices[:, 1]

        # normalize image
        if self.image_normalizer:
            mean, std = self.image_normalizer
            mean = np.asarray(mean, dtype=np.float32)
            std = np.asarray(std, dtype=np.float32)
            image = (image - mean) / std

        out_dict['img'] = np.moveaxis(image, -1, 0) # shape C, H, W

        # 3D data augmentation and scaling from points to voxel indices
        # Kitti lidar coordinates: x (front), y (left), z (up)
        coords = augment_and_scale_3d(points, self.scale, self.full_scale, noisy_rot=self.noisy_rot,
                                      flip_y=self.flip_y, rot_z=self.rot_z, transl=self.transl)

        # cast to integer
        coords = coords.astype(np.int64)
    
        # coords.min(1) -> minimum coordinate for each point, shape (N,)
        # coords.max(1) -> max coordinate for each point. shape (N,)
        # only use voxels inside receptive field
        voxel_valid_idxs = (coords.min(1) >= 0) * (coords.max(1) < self.full_scale)
        voxel_coords = coords[voxel_valid_idxs]
        voxel_feats = feats[voxel_valid_idxs]
        voxel_seg_label = seg_label[voxel_valid_idxs]
        voxel_img_indices = img_indices[voxel_valid_idxs]

        sparse_unique_inds, _, sparse_inverse_map = sparse_quantize(voxel_coords, voxel_feats, voxel_seg_label, return_index=True, return_invs=True)
        # print(type(inverse_map),"inverse map: ", inverse_map.shape, " coords: ", coords.shape, " inds ", inds.shape)
        #import pdb; pdb.set_trace();
        out_dict["coords"] = voxel_coords[sparse_unique_inds]
        out_dict['feats'] = voxel_feats[sparse_unique_inds]
        out_dict['seg_label'] = voxel_seg_label[sparse_unique_inds]
        out_dict['img_indices'] = voxel_img_indices[sparse_unique_inds]
        # out_dict["lidar"] = SparseTensor(coords=coords[inds], feats=feats[inds])
        # out_dict['seg_label'] = SparseTensor(coords=coords[inds], feats=seg_label[inds])
        # out_dict['img_indices'] = img_indices[inds].tolist()#SparseTensor(coords=coords[inds], feats=img_indices[inds])
        # out_dict["inverse_map"] = SparseTensor(coords=coords[inds], feats=inverse_map) 

        if self.output_orig:
            out_dict.update({
                'orig_seg_label': seg_label,
                'sparse_orig_points_idx': voxel_valid_idxs[sparse_unique_inds],
                "inverse_map": sparse_inverse_map
            })

        return out_dict


def compute_class_weights():
    preprocess_dir = '/home/user/SemanticKitti/preprocessed'
    split = ('train',)
    dataset = SemanticKITTIBase(split,
                                preprocess_dir,
                                )
    # compute points per class over whole dataset
    num_classes = len(np.unique(list(dataset.semantic_kitti_config_dict["learning_map"].values())))
    points_per_class = np.zeros(num_classes, int)
    for i, data_path in enumerate(dataset.data_paths):
        print('{}/{}'.format(i, len(dataset)))
        # labels = dataset.label_mapping[data['seg_labels']]
        with open(data_path, 'rb') as data_file:
            data = pickle.load(data_file)
        labels = dataset.map_label(data['seg_labels'])
        points_per_class += np.bincount(labels, minlength=num_classes)

    points_per_class = points_per_class[1:] # ignore zero class
    print(points_per_class)
    class_weights = np.log(5 * points_per_class.sum() / points_per_class)
    print('log smoothed class weights: ', class_weights / class_weights.min())


if __name__ == '__main__':
    # test_SemanticKITTISCN()
    compute_class_weights()
