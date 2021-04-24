# FusionTransformer

Official code for the paper.

## Paper
## Preparation
### Prerequisites

### Installation

### Datasets
#### NuScenes
Please download the Full dataset (v1.0) from the [NuScenes website](https://www.nuscenes.org) and extract it.

You need to perform preprocessing to generate the data for FusionTransformer first.
The preprocessing subsamples the 360° LiDAR point cloud to only keep the points that project into
the front camera image. It also generates the point-wise segmentation labels using
the 3D objects by checking which points lie inside the 3D boxes. 
All information will be stored in a pickle file (except the images which will be 
read frame by frame by the dataloader during training).

Please edit the script `FusionTransformer/data/nuscenes/preprocess.py` as follows and then run it.
* `root_dir` should point to the root directory of the NuScenes dataset
* `out_dir` should point to the desired output directory to store the pickle files

#### SemanticKITTI
Please download the files from the [SemanticKITTI website](http://semantic-kitti.org/dataset.html) and
additionally the [color data](http://www.cvlibs.net/download.php?file=data_odometry_color.zip)
from the [Kitti Odometry website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php). Extract
everything into the same folder.

Similar to NuScenes preprocessing, we save all points that project into the front camera image as well
as the segmentation labels to a pickle file.

Please edit the script `FusionTransformer/data/semantic_kitti/preprocess.py` as follows and then run it.
* `root_dir` should point to the root directory of the SemanticKITTI dataset
* `out_dir` should point to the desired output directory to store the pickle files

## Training

### Baseline


## Testing


## Model Zoo

## Acknowledgements
Note that this code is forked from he [xmuda](https://github.com/maxjaritz/xmuda") repo.
```
@inproceedings{jaritz2019xmuda,
	title={{xmuda}: Cross-Modal Unsupervised Domain Adaptation for {3D} Semantic Segmentation},
	author={Jaritz, Maximilian and Vu, Tuan-Hung and de Charette, Raoul and Wirbel, Emilie and P{\'e}rez, Patrick},
	booktitle={CVPR},
	year={2020}
}
```
## License
this repo is released under the [Apache 2.0 license](./LICENSE).