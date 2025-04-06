# Overview
Wanted to get some practical experience with specifically implementing a Visual Pose-Graph SLAM pipeline.

# Dataset
Built using TUM Computer Vision Group's [RGB-D dataset](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) which includes ground truth information.
Store this in the root directory.

## Useful tools

The dataset provides many useful additional tools, which can be downloaded [here](https://svncvpr.in.tum.de/cvpr-ros-pkg/trunk/rgbd_benchmark/rgbd_benchmark_tools/src/rgbd_benchmark_tools/).

### `associate.py`
The RGB and depth images are captured asynchronously, so it's essential to associate them correctly. 
The [dataset provides](https://svncvpr.in.tum.de/cvpr-ros-pkg/trunk/rgbd_benchmark/rgbd_benchmark_tools/src/rgbd_benchmark_tools/) a Python script called `associate.py` to match RGB and depth images based on timestamps.
The original `associate.py` has been modified particularly at lines `86` and `87` to solve an issue.

Inside the `rgbd_dataset_...` directory(s) run:
```
python3 ../tools/associate.py rgb.txt depth.txt > associations.txt
```

This will generate an associations.txt file linking corresponding RGB and depth images.

### Optional:
### `add_pointclouds_to_bagfile.py`
### `evaluate_ate.py`
### `evaluate_rpe.py`
### `generate_pointcloud.py`
### `generate_registered_pointcloud.py`
### `plot_trajectory_into_image.py`

## Requirements
```
pip install open3d numpy gtsam matplotlib
```