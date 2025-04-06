import numpy as np
import os

np.float = float
np.int = int
np.bool = bool
np.object = object
np.complex = complex

import open3d as o3d


def read_associations(assoc_path):
    rgbd_pairs = []
    with open(assoc_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                rgbd_pairs.append((parts[0], parts[1], parts[2], parts[3]))
    return rgbd_pairs

def make_rgbd_image(color_path, depth_path):
    color = o3d.io.read_image(color_path)
    depth = o3d.io.read_image(depth_path)
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, convert_rgb_to_intensity=False)

def extract_odometry(dataset_path, assoc_file, max_pairs=100):
    assoc = read_associations(assoc_file)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    
    poses = [np.eye(4)]
    
    for i in range(1, min(len(assoc), max_pairs)):
        rgb1 = os.path.join(dataset_path, assoc[i-1][1])
        d1   = os.path.join(dataset_path, assoc[i-1][3])
        rgb2 = os.path.join(dataset_path, assoc[i][1])
        d2   = os.path.join(dataset_path, assoc[i][3])
        
        rgbd1 = make_rgbd_image(rgb1, d1)
        rgbd2 = make_rgbd_image(rgb2, d2)

        success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
            rgbd1, rgbd2, intrinsic, np.identity(4),
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm())
        
        if success:
            new_pose = poses[-1] @ trans
            poses.append(new_pose)
        else:
            poses.append(poses[-1])  # fallback to previous pose

    return poses
