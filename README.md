# Overview
Wanted to get some practical experience with specifically implementing a Visual Pose-Graph SLAM pipeline.

# Dataset
Built using TUM Computer Vision Group's [RGB-D dataset](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) which includes ground truth information.
Store this in the root directory.

## `associate.py`
The RGB and depth images are captured asynchronously, so it's essential to associate them correctly. 
The [dataset provides](https://svncvpr.in.tum.de/cvpr-ros-pkg/trunk/rgbd_benchmark/rgbd_benchmark_tools/src/rgbd_benchmark_tools/) a Python script called `associate.py` to match RGB and depth images based on timestamps.
The original `associate.py` has been modified particularly at lines `86` and `87` to solve an issue.

Inside the `rgbd_dataset_...` directory(s) run:
```
python3 ../associate.py rgb.txt depth.txt > associations.txt
```

This will generate an associations.txt file linking corresponding RGB and depth images.

