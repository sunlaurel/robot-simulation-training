from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import random
import numpy as np
import torch
import math
from sklearn.model_selection import train_test_split

""" Constants """
RADIUS = 0.75


def GenTrainTestGeneratedDatasets(csv_path, past_steps, future_steps):
    csv_data = pd.read_csv(csv_path)
    person_ids = csv_data["id"].unique()
    train_ids, test_ids = train_test_split(person_ids, test_size=0.2, random_state=2244)
    train_dataset = GeneratedTrajectoryDataset(
        csv_data, train_ids, N_past=past_steps, N_future=future_steps
    )
    test_dataset = GeneratedTrajectoryDataset(
        csv_data, test_ids, N_past=past_steps, N_future=future_steps
    )
    return train_dataset, test_dataset


class GeneratedTrajectoryDataset(Dataset):

    def __init__(self, csv_data, split_ids, N_past, N_future):
        self.N_past = N_past
        self.N_future = N_future
        self.person_ids = csv_data["id"].unique()
        self.position_data = {}  # mapping from id to trajectory
        self.velocity_data = {}  # mapping from id to trajectory
        self.len = len(csv_data)
        self.person_ids = split_ids

        for person_id in tqdm(
            self.person_ids, total=len(self.person_ids), dynamic_ncols=True
        ):
            trajdata = csv_data[csv_data["id"] == person_id].sort_values(by="time")
            if len(trajdata) <= self.N_past + self.N_future:
                continue
            self.position_data[person_id] = trajdata[["x", "y"]].to_numpy().T  # 4 x N

        self.person_ids = list(
            self.position_data.keys()
        )  # remove people with short trajectories

    def __len__(self):
        # The dataset length is the number of rows in the CSV file
        return self.len

    def __getitem__(self, idx):
        initial_offset = random.uniform(
            -0.5, 0.5
        )  # changing the robot's starting position

        epsilon = 5e-02

        # 0. X -- Undo my changes so the robot goes back to going at different velocities at different times
        # 1. Include the velocity at each step into the network input (so the robot can know what velocity its going)
        # 2. Retrain the network and let us know the new test error and behavior
        # 3. Visualize 10-20 training examples along with the prediction so we can all see together what kinds of inputs
        #    the network is getting correct and what kinds its getting wrong

        # TODO: play with the initial x and y offsets and x and y velocities of the robot
        random_person_id = int(random.choice(self.person_ids))
        X = self.position_data[random_person_id]
        random_frame = int(random.randint(self.N_past, X.shape[1] - 1 - self.N_future))

        # Determine the indices for past and future points
        X_past = X[:, random_frame - self.N_past + 1 : random_frame + 1].copy()
        X_future = X[:, random_frame + 1 : random_frame + 1 + self.N_future].copy()

        present_tangent = X_past[:, -1] - X_past[:, -2]

        #########  Constraining velocities to be same direction as person  ##########
        x_vel = random.uniform(
            0, 3 * 0.12
        )  # max speed would be 3 m/s * 0.12 s/frame = 0.36 m/frame
        x_vel = -x_vel if present_tangent[0] * x_vel < 0 else x_vel
        y_vel = random.uniform(
            0, math.sqrt((3 * 0.12) ** 2 - x_vel**2)
        )  # constraining the y velocity so that the speed of the robot is max 3 m/s
        y_vel = -y_vel if present_tangent[1] * y_vel < 0 else y_vel
        # mag = math.sqrt(x_vel **2 + y_vel**2)/0.12
        # x_vel /= mag
        # y_vel /= mag

        ##########  Generating the path for the robot ########
        starting_pos = (
            X_past[:, 0] + initial_offset
        )  # starting points for the generated line

        generated_traj_x = np.linspace(
            starting_pos[0], starting_pos[0] + (self.N_past - 1) * x_vel, self.N_past
        )

        generated_traj_y = np.linspace(
            starting_pos[1], starting_pos[1] + (self.N_past - 1) * y_vel, self.N_past
        )

        generated_traj = np.array([generated_traj_x, generated_traj_y])

        #########  Shifting it to the robot's coordinate frame  ##########
        curr_pos = generated_traj[:, -1].copy()
        generated_traj[0] -= generated_traj[0, -1]
        generated_traj[1] -= generated_traj[1, -1]

        X_past[0] -= curr_pos[0]
        X_past[1] -= curr_pos[1]
        X_future[0] -= curr_pos[0]
        X_future[1] -= curr_pos[1]
        past_relative_vectors = X_past - generated_traj

        # getting the velocity at each step
        # breakpoint()
        X_vel = generated_traj[:, 1:] - generated_traj[:, :-1]
        X_vel = np.column_stack((X_vel, X_vel[:, -1].copy()))
        input_vectors = np.vstack((past_relative_vectors, X_vel))

        ##########  Calculating the target position  ##########
        X = -past_relative_vectors[:, -1]
        present_perp = np.array(
            [X_past[1, -1] - X_past[1, -2], -(X_past[0, -1] - X_past[0, -2])]
        )

        future_perp = np.array(
            [
                X_future[1, -1] - X_future[1, -2],
                -(X_future[0, -1] - X_future[0, -2]),
            ]
        )

        if np.linalg.norm(present_perp) <= epsilon:
            future_pos = X / np.linalg.norm(X) * RADIUS
        else:
            future_perp /= np.linalg.norm(future_perp)
            present_perp /= np.linalg.norm(present_perp)
            t = X @ present_perp
            offset = t * future_perp
            offset = offset / np.linalg.norm(offset) * RADIUS
            offset = future_perp  # trying just using the future position as the orthogonal vector of the 2 future positions
            future_pos = X_future[:2, -1] + offset

        ###########  Graphing generated trajectory ###########
        # plt.figure(figsize=(12, 12))
        # plt.xlim(-5, 5)
        # plt.ylim(-5, 5)
        # plt.scatter(
        #     X_past[0],
        #     X_past[1],
        #     label="Original Past Trajectory",
        # )
        # plt.scatter(X_future[0], X_future[1], label="Future Trajectory")
        # plt.scatter(
        #     generated_traj[0], generated_traj[1], label="Generated Robot Trajectory"
        # )

        # for i in range(len(past_relative_vectors[0])):
        #     plt.plot(
        #         [X_past[0][i], X_past[0][i] - past_relative_vectors[0][i]],
        #         [X_past[1][i], X_past[1][i] - past_relative_vectors[1][i]],
        #     )

        # # plt.quiver(
        # #     generated_traj[0],
        # #     generated_traj[1],
        # #     past_relative_vectors[0],
        # #     past_relative_vectors[1],
        # #     label="Past Relative Vectors",
        # # )

        # plt.scatter(
        #     future_pos[0],
        #     future_pos[1],
        #     color="red",
        #     label="Future Position",
        # )
        # plt.legend()
        # plt.title("Trajectories")
        # plt.xlabel("x")
        # plt.ylabel("y")
        # plt.show()

        return (
            input_vectors,
            future_pos,
            X_past,
            X_future,
            generated_traj,
        )


def GenTrainTestDatasets(csv_path, past_steps, future_steps):
    csv_data = pd.read_csv(csv_path)
    person_ids = csv_data["id"].unique()
    train_ids, test_ids = train_test_split(person_ids, test_size=0.2, random_state=2244)
    train_dataset = TrajectoryDataset(
        csv_data, train_ids, N_past=past_steps, N_future=future_steps
    )
    test_dataset = TrajectoryDataset(
        csv_data, test_ids, N_past=past_steps, N_future=future_steps
    )
    return train_dataset, test_dataset


class TrajectoryDataset(Dataset):

    def __init__(self, csv_data, split_ids, N_past, N_future):
        self.N_past = N_past
        self.N_future = N_future
        self.person_ids = csv_data["id"].unique()
        self.position_data = {}  # mapping from id to trajectory
        # self.velocity_data = {}  # mapping from id to trajectory
        self.len = len(csv_data)

        self.person_ids = split_ids
        self.position_data = {}  # mapping from id to trajectory
        # self.velocity_data = {} # mapping from id to trajectory
        for person_id in tqdm(
            self.person_ids, total=len(self.person_ids), dynamic_ncols=True
        ):
            trajdata = csv_data[csv_data["id"] == person_id].sort_values(by="time")
            if (
                len(trajdata) <= self.N_past + self.N_future
            ):  # look at this again later to see if it should be <= or just <
                continue
            self.position_data[person_id] = trajdata[["x", "y"]].to_numpy().T  # 2 x N
            # self.velocity_data[person_id] = trajdata[["v_x","v_y"]].to_numpy().T # 2 x N

        self.person_ids = list(
            self.position_data.keys()
        )  # remove people with short trajectories

    def __len__(self):
        # The dataset length is the number of rows in the CSV file
        return self.len

    def __getitem__(self, idx):
        random_person_id = int(random.choice(self.person_ids))
        X = self.position_data[random_person_id]
        # V = self.velocity_data[random_person_id]
        random_frame = int(random.randint(self.N_past, X.shape[1] - 1 - self.N_future))

        # Determine the indices for past and future points
        X_past = X[:, random_frame - self.N_past + 1 : random_frame + 1]
        X_future = X[:, random_frame + 1 : random_frame + 1 + self.N_future]

        return X_past, X_future


def plot_trajectory(x_past, x_future, v_past=None, v_future=None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    x_past_coords = x_past[0, :].numpy()
    y_past_coords = x_past[1, :].numpy()
    x_future_coords = x_future[0, :].numpy()
    y_future_coords = x_future[1, :].numpy()

    dt = 0.12
    vx_past_coords = v_past[0, :].numpy()
    vy_past_coords = v_past[1, :].numpy()
    vx_future_coords = v_future[0, :].numpy()
    vy_future_coords = v_future[1, :].numpy()

    axes[0].plot(x_past_coords, y_past_coords, marker="o", linestyle="-", color="r")
    axes[0].plot(x_future_coords, y_future_coords, marker="o", linestyle="-", color="b")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("Positions")

    axes[1].plot(x_past_coords, y_past_coords, marker="o", linestyle="-", color="k")
    axes[1].plot(x_future_coords, y_future_coords, marker="o", linestyle="-", color="k")
    axes[1].quiver(
        x_past_coords,
        y_past_coords,
        vx_past_coords,
        vy_past_coords,
        color="r",
        angles="xy",
        scale_units="xy",
        scale=1 / dt,
    )
    axes[1].quiver(
        x_future_coords,
        y_future_coords,
        vx_future_coords,
        vy_future_coords,
        color="b",
        angles="xy",
        scale_units="xy",
        scale=1 / dt,
    )

    axes[1].set_title("Velocity Vectors")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].grid(True)
    plt.show()
