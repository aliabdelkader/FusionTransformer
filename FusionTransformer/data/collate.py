import torch
from functools import partial
from torchsparse.sparse_tensor import SparseTensor
import numpy as np

def collate_scn_base(input_dict_list, output_orig, output_image=True):
    """
    Custom collate function for SCN. The batch size is always 1,
    but the batch indices are appended to the locations.
    :param input_dict_list: a list of dicts from the dataloader
    :param output_orig: whether to output original point cloud/labels/indices
    :param output_image: whether to output images
    :return: Collated data batch as dict
    """
    locs=[]
    feats=[]
    labels=[]
    seq = []
    filename = []
    voxel_coords = []


    if output_image:
        imgs = []
        img_idxs = []

    if output_orig:
        orig_seg_label = []
        sparse_orig_points_idx = []
        inverse_maps = []

    # output_pselab = 'pseudo_label_2d' in input_dict_list[0].keys()
    # if output_pselab:
    #     pseudo_label_2d = []
    #     pseudo_label_3d = []

    for idx, input_dict in enumerate(input_dict_list):
        voxel_coords.append(input_dict['voxel_coords'])
        coords = torch.from_numpy(input_dict['coords'])
        # add batch idx to coordinates
        batch_idxs = torch.LongTensor(coords.shape[0], 1).fill_(idx)
        locs.append(torch.cat([coords, batch_idxs], 1)) # locs = [coords, batch]
        feats.append(torch.from_numpy(input_dict['feats']))

        if 'seg_label' in input_dict.keys():
            labels.append(torch.from_numpy(input_dict['seg_label']))

        if output_image:
            imgs.append(torch.from_numpy(input_dict['img']))
            img_idxs.append(input_dict['img_indices'])

        if output_orig:
            orig_seg_label.append(input_dict['orig_seg_label'])
            sparse_orig_points_idx.append(input_dict['sparse_orig_points_idx'])
            inverse_maps.append(input_dict["inverse_map"])
        
        seq.append(input_dict["seq"])
        filename.append(input_dict["filename"] )
        # if output_pselab:
        #     pseudo_label_2d.append(torch.from_numpy(input_dict['pseudo_label_2d']))
        #     if input_dict['pseudo_label_3d'] is not None:
        #         pseudo_label_3d.append(torch.from_numpy(input_dict['pseudo_label_3d']))

    locs = torch.cat(locs, 0)
    feats = torch.cat(feats, 0)

    out_dict = {'lidar': SparseTensor(coords=locs, feats=feats)}
    if labels:
        labels = torch.cat(labels, 0)
        out_dict['seg_label'] = labels
    if output_image:
        out_dict['img'] = torch.stack(imgs)
        out_dict['img_indices'] = img_idxs
        
    if output_orig:
        out_dict['orig_seg_label'] = orig_seg_label
        out_dict['sparse_orig_points_idx'] = sparse_orig_points_idx
        out_dict["inverse_map"] = inverse_maps
    
    out_dict["seq"] = seq
    out_dict["filename"] = filename
    out_dict['voxel_coords'] = voxel_coords
    # if output_pselab:
    #     out_dict['pseudo_label_2d'] = torch.cat(pseudo_label_2d, 0)
    #     out_dict['pseudo_label_3d'] = torch.cat(pseudo_label_3d, 0) if pseudo_label_3d else pseudo_label_3d
    return out_dict


def get_collate_scn(is_train):
    return partial(collate_scn_base,
                   output_orig=not is_train,
                   )
