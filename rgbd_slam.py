
import os
import cv2
import numpy as np
np.float = float
np.int = int
np.bool = bool
np.object = object
np.complex = complex
import open3d as o3d
import gtsam
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

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
    logging.info("Reading associations...")
    assoc = read_associations(assoc_file)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

    poses = [np.eye(4)]
    logging.info(f"Computing odometry for {min(len(assoc), max_pairs)} frames...")

    for i in tqdm(range(1, min(len(assoc), max_pairs)), desc="Computing Odometry"):
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
            logging.warning(f"Odometry failed between frames {i-1} and {i}")
            poses.append(poses[-1])  # fallback to previous pose

    return poses, assoc

def to_pose3(T):
    R = gtsam.Rot3(T[:3, :3])
    t = gtsam.Point3(T[0, 3], T[1, 3], T[2, 3])
    return gtsam.Pose3(R, t)

def build_graph(poses):
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6]*6))
    odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05]*6))

    graph.add(gtsam.PriorFactorPose3(0, to_pose3(poses[0]), prior_noise))
    initial.insert(0, to_pose3(poses[0]))

    for i in range(1, len(poses)):
        T_prev = poses[i-1]
        T_curr = poses[i]
        T_rel = np.linalg.inv(T_prev) @ T_curr
        pose_rel = to_pose3(T_rel)
        graph.add(gtsam.BetweenFactorPose3(i-1, i, pose_rel, odom_noise))
        initial.insert(i, to_pose3(poses[i]))

    return graph, initial

def optimize_graph(graph, initial):
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial)
    result = optimizer.optimize()
    return result

def animate_results(poses, dataset_path, assoc, result):
    fig = plt.figure(figsize=(10, 5))
    ax_traj = fig.add_subplot(121, projection='3d')
    ax_img = fig.add_subplot(122)
    ax_img.axis('off')

    xs, ys, zs = [], [], []
    images = []

    logging.info("Preloading images...")
    for _, rgb_file, _, _ in tqdm(assoc[:len(poses)], desc="Loading RGB frames"):
        rgb_path = os.path.join(dataset_path, rgb_file)
        img = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        images.append(img)

    img_display = ax_img.imshow(images[0])
    traj_line, = ax_traj.plot([], [], [], 'b-')
    start_dot = ax_traj.scatter([], [], [], c='g')
    end_dot = ax_traj.scatter([], [], [], c='r')

    def update(i):
        pose = result.atPose3(i)
        t = pose.translation()
        xs.append(t[0])
        ys.append(t[1])
        zs.append(t[2])

        ax_traj.cla()
        ax_traj.set_title("Optimized Trajectory")
        ax_traj.set_xlim(-2, 2)
        ax_traj.set_ylim(-2, 2)
        ax_traj.set_zlim(0, 2)
        ax_traj.plot(xs, ys, zs, 'b-')
        ax_traj.scatter(xs[0], ys[0], zs[0], c='g')
        ax_traj.scatter(xs[-1], ys[-1], zs[-1], c='r')

        img_display.set_data(images[i])

    ani = animation.FuncAnimation(fig, update, frames=len(poses), interval=50)
    plt.show()

if __name__ == "__main__":
    dataset_path = "rgbd_dataset_freiburg1_xyz"  # change if needed
    assoc_file = f"{dataset_path}/associations.txt"

    poses, assoc = extract_odometry(dataset_path, assoc_file, max_pairs=100)
    logging.info("Odometry extraction complete.")

    graph, initial = build_graph(poses)
    logging.info("Factor graph constructed.")

    result = optimize_graph(graph, initial)
    logging.info("Optimization complete. Launching animation...")

    animate_results(poses, dataset_path, assoc, result)