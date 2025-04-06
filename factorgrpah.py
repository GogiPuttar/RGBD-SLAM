import gtsam
import numpy as np
import matplotlib.pyplot as plt
from odometry import extract_odometry

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2

def to_pose3(T):
    R = gtsam.Rot3(T[:3, :3])
    t = gtsam.Point3(T[0, 3], T[1, 3], T[2, 3])
    return gtsam.Pose3(R, t)

def build_graph(poses):
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    # Noise models
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6]*6))
    odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05]*6))  # tune as needed

    # Add prior on first pose
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

def optimize_and_plot(graph, initial):
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial)
    result = optimizer.optimize()

    # Plot
    xs = []
    ys = []
    zs = []

    for i in range(initial.size()):
        pose = result.atPose3(i)
        t = pose.translation()
        xs.append(t[0])
        ys.append(t[1])
        zs.append(t[2])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs, ys, zs, label="Optimized trajectory", color="blue")
    ax.scatter(xs[0], ys[0], zs[0], color="green", label="Start")
    ax.scatter(xs[-1], ys[-1], zs[-1], color="red", label="End")
    ax.legend()
    plt.show()

def animate_results(poses, dataset_path, assoc, result):
    fig = plt.figure(figsize=(10, 5))
    ax_traj = fig.add_subplot(121, projection='3d')
    ax_img = fig.add_subplot(122)
    ax_img.axis('off')

    xs, ys, zs = [], [], []
    img_plot = None

    def update(i):
        ax_traj.cla()
        ax_traj.set_title("Optimized Trajectory")
        ax_traj.set_xlim(-2, 2)
        ax_traj.set_ylim(-2, 2)
        ax_traj.set_zlim(0, 2)

        pose = result.atPose3(i)
        t = pose.translation()
        xs.append(t[0])
        ys.append(t[1])
        zs.append(t[2])

        ax_traj.plot(xs, ys, zs, 'b-')
        ax_traj.scatter(xs[0], ys[0], zs[0], c='g')
        ax_traj.scatter(xs[-1], ys[-1], zs[-1], c='r')

        rgb_path = os.path.join(dataset_path, assoc[i][1])
        img = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        ax_img.imshow(img)

    ani = animation.FuncAnimation(fig, update, frames=len(poses), interval=100)
    plt.show()

if __name__ == "__main__":

    dataset_path = "rgbd_dataset_freiburg1_xyz"
    assoc_file = f"{dataset_path}/associations.txt"

    poses = extract_odometry(dataset_path, assoc_file, max_pairs=100)

    graph, initial = build_graph(poses)

    optimize_and_plot(graph, initial)

    animate_results(poses, dataset_path, assoc, result)