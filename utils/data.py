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

###########################################################
##   Overview                                            ##
##   - generates robot trajectories for training         ##
##   - also has functions that convert from world to     ##
##     robot frame                                       ##
###########################################################

""" Constants """
RADIUS = 0.75  # how far the future position should be from the person


def ProcessPast(past_relative_vectors, robot_velocities_r):
    input_vectors = np.vstack((past_relative_vectors, robot_velocities_r))

    return input_vectors


def ConvertAbsoluteToRobotFrame(X_past, R_past, heading, future_pos):
    R_tangent = heading  # R_past[:, 1:] - R_past[:, :-1]
    R_tangent = np.column_stack((np.copy(R_tangent[:, 0]), R_tangent))
    R_tangent /= np.linalg.norm(R_tangent, axis=0)
    R_perpendicular = np.array([-R_tangent[1], R_tangent[0]])

    # fig, ax = plt.subplots(figsize=(10, 9))

    # TODO: make more efficient without for loop
    breakpoint()
    relative_vectors_w = X_past - R_past
    relative_future_position_w = torch.tensor(future_pos[:, None]) - torch.tensor(R_past)
    relative_vectors_r_lst = []
    robot_velocity_r_lst = []
    relative_velocities_w = np.zeros_like(R_past)
    # relative_velocities_w[1:] = R_past[1:] - R_past[0:-1]
    # relative_velocities_w[0] = relative_velocities_w[1]
    relative_velocities_w[:, 1:] = R_past[:, 1:] - R_past[:, :-1]
    relative_velocities_w[:, 0] = relative_velocities_w[:, 1]
    relative_velocities_w /= 0.12

    for i in range(0, X_past.shape[1]):
        ### debugging ###
        # ax.quiver(
        #     R_past[0, i], R_past[1, i], R_tangent[0, i], R_tangent[1, i], color="red"
        # )
        # ax.quiver(
        #     R_past[0, i], R_past[1, i], R_perpendicular[0, i], R_perpendicular[1, i], color="green"
        # )

        current_relative_vector_w = relative_vectors_w[:, i]
        current_relative_velocity_w = relative_velocities_w[:, i]
        current_relative_future_position_w = relative_future_position_w[:, i]

        # relative_vectors = X_past[:, i - 1:i + 1] - R_past[:, i - 1: i + 1]

        rotation_matrix_w_r = torch.tensor([R_tangent[:, i], R_perpendicular[:, i]])

        R_tangent_r = rotation_matrix_w_r @ R_tangent
        R_perpendicular_r = rotation_matrix_w_r @ R_perpendicular
        # relative_vectors = rotation_matrix @ relative_vectors

        relative_vectors_r = rotation_matrix_w_r @ relative_vectors_w
        # relative_velocities_r = rotation_matrix_w_r @ relative_velocities_w
        current_relative_vector_r = relative_vectors_r[:, i]
        relative_future_position_r = (
            rotation_matrix_w_r @ current_relative_future_position_w
        )
        current_relative_velocity_r = rotation_matrix_w_r @ current_relative_velocity_w

        # adding to the list
        relative_vectors_r_lst.append(current_relative_vector_r)
        robot_velocity_r_lst.append(current_relative_velocity_r)
        future_pos_r = relative_future_position_r

        # ConvertRobotFrameToAbsolute(, future_pos_r)

        # # # # if __DEBUG__:
        # if i == X_past.shape[1] - 1:
        #     fig, ax = plt.subplots(1, 2, figsize=(12, 9))
        #     ax[0].scatter(future_pos[0], future_pos[1])
        #     ax[0].arrow(R_past[0,i], R_past[1,i], current_relative_velocity_w[0], current_relative_velocity_w[1], color="orange")
        #     ax[0].arrow(R_past[0,i], R_past[1,i], R_tangent[0, i], R_tangent[1, i], color="red")
        #     ax[0].arrow(R_past[0,i], R_past[1,i], R_perpendicular[0, i], R_perpendicular[1, i], color="green")
        #     ax[0].arrow(R_past[0,i], R_past[1,i], current_relative_vector_w[0], current_relative_vector_w[1], color="gray")
        #     ax[0].arrow(R_past[0,i], R_past[1,i], current_relative_future_position_w[0], current_relative_future_position_w[1], color="purple")
        #     ax[0].scatter(R_past[0], R_past[1], color="black")
        #     ax[0].scatter(X_past[0], X_past[1], color="gray")

        #     # ax[1].arrow(0, 0, current_relative_velocity_r[0], current_relative_velocity_r[1], color="orange")
        #     # ax[1].arrow(0, 0, R_tangent_r[0, i], R_tangent_r[1, i], color="red")
        #     # ax[1].arrow(0, 0, R_perpendicular_r[0, i], R_perpendicular_r[1, i], color="green")
        #     # ax[1].arrow(0, 0, current_relative_vector_r[0], current_relative_vector_r[1], color="gray")
        #     # ax[1].arrow(0, 0, relative_future_position_r[0], relative_future_position_r[1], color="purple")
        #     ax[1].arrow(0, 0, current_relative_velocity_r[0], current_relative_velocity_r[1], color="orange")
        #     ax[1].arrow(0, 0, R_tangent_r[0, i], R_tangent_r[1, i], color="red")
        #     ax[1].arrow(0, 0, R_perpendicular_r[0, i], R_perpendicular_r[1, i], color="green")
        #     ax[1].arrow(0, 0, current_relative_vector_r[0], current_relative_vector_r[1], color="gray")
        #     ax[1].arrow(0, 0, relative_future_position_r[0], relative_future_position_r[1], color="purple")

        #     ax[0].set_aspect("equal")
        #     ax[1].set_aspect("equal")
        #     ax[0].set_title("Graph in World Coordinates")
        #     ax[1].set_title("Graph in Robot Coordinates")
        #     print(relative_future_position_r)
        #     plt.show()

    ## debug ##
    # ax.scatter(X_past[0], X_past[1], color="black", label="Person")
    # ax.scatter(R_past[0], R_past[1], color="gray", label="Robot")
    # ax.scatter(future_pos[0], future_pos[1], color="purple")
    # ax.legend()
    # ax.set_aspect("equal")
    # plt.show()
    return (
        np.array(relative_vectors_r_lst),
        np.array(robot_velocity_r_lst),
        relative_future_position_r,
    )


def ConvertRobotFrameToAbsolute(heading, future_pos_r):
    R_tangent = heading
    R_tangent /= np.linalg.norm(R_tangent)
    R_perpendicular = np.array([-R_tangent[1], R_tangent[0]])

    rotation_matrix_w_r = torch.stack(
        [torch.tensor(R_tangent), torch.tensor(R_perpendicular)]
    ).float()
    rotation_matrix_r_w = rotation_matrix_w_r.T
    future_pos_w = rotation_matrix_r_w @ future_pos_r.float()

    # fig, ax = plt.subplots(1, 2, figsize=(12, 9))
    # ax[0].arrow(0, 0, 1, 0, color="red")
    # ax[0].arrow(0, 0, 0, 1, color="green")
    # ax[0].arrow(0, 0, future_pos_r[0], future_pos_r[1], color="gray")

    # ax[1].arrow(0, 0, R_tangent[0], R_tangent[1], color="red")
    # ax[1].arrow(0, 0, R_perpendicular[0], R_perpendicular[1], color="green")
    # ax[1].arrow(0, 0, future_pos_w[0], future_pos_w[1], color="gray")

    # ax[0].set_aspect("equal")
    # ax[1].set_aspect("equal")
    # plt.show()

    return future_pos_w


########################  Dataset for generating robot trajectories  ########################
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
        random.seed(1776)
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
        initial_offset = random.uniform(-4, 4)  # changing the robot's starting position
        epsilon = 5e-02

        random_person_id = int(random.choice(self.person_ids))
        data = self.position_data[random_person_id]
        random_frame = int(
            random.randint(self.N_past, data.shape[1] - 1 - self.N_future)
        )

        # Determine the indices for past and future points
        X_past = data[:, random_frame - self.N_past + 1 : random_frame + 1].copy()
        X_future = data[:, random_frame + 1 : random_frame + 1 + self.N_future].copy()

        ###############  Generating noise for the robot ################
        sigma = 0.15
        N = np.random.rand(*X_past.shape) * sigma
        X_past = torch.tensor(X_past)
        X_past += torch.tensor(N, dtype=torch.float)

        ###############  Generating random velocities for the robot  ##############
        v = np.random.uniform(-1, 1, size=2)
        v /= np.linalg.norm(v)
        s = random.uniform(0, 2)
        v *= s

        ###############  Generating random curvatures for the robot  ##############
        angle = random.uniform(-math.pi, math.pi)  # how much of the arc is used
        radius = random.uniform(0, 2)  # generating a random radius from 0 - 2m
        theta = np.linspace(0, angle, self.N_past)

        #################  Generating the path for the robot  ##############
        starting_pos = np.array(
            X_past[:, 0] + initial_offset
        )  # starting points for the generated line

        # Alternating between curved or straight path
        curved_or_straight_path = bool(random.randint(0, 1))
        copy_person = bool(random.randint(1, 10))  # copy_person lets robot copy the person's movement
        if copy_person == 1:
            # shifting human trajectory
            offset = np.random.rand(2) * 8 - 4  # limiting the offset to -4 to 4
            offset = offset.reshape(2, 1)
            generated_traj = X_past + offset
            # plt.scatter(generated_traj[0], generated_traj[1])
            # plt.scatter(X_past[0], X_past[1])
            # plt.show()
        else:   
            if curved_or_straight_path:
                generated_traj = np.empty((2, self.N_past))
                generated_traj[:, 0] = starting_pos

                for i in range(1, self.N_past):
                    generated_traj[:, i] = generated_traj[:, i - 1] + v * 0.12
            else:
                generated_traj = np.array(
                    [
                        starting_pos[0] + radius * np.cos(theta),
                        starting_pos[1] + radius * np.sin(theta),
                    ]
                )

        #################  Calculating the target position  ################
        X = generated_traj[:, -1] - np.array(X_past[:, -1])
        present_perp = np.array(
            [X_past[1, -1] - X_past[1, -2], -(X_past[0, -1] - X_past[0, -2])]
        )

        future_perp = np.array(
            [
                X_future[1, -1] - X_future[1, -2],
                -(X_future[0, -1] - X_future[0, -2]),
            ]
        )

        if (
            np.linalg.norm(present_perp) <= epsilon
            or np.linalg.norm(future_perp) <= epsilon
        ):
            ###############  Generating the case when the person is standing still  ################
            if (
                np.linalg.norm(X) <= RADIUS
            ):  # if the robot is within 0.75m of the person, then the future position is the robot's current position
                future_pos = (
                    0.9 * torch.tensor(generated_traj[:, -1]) + 0.1 * X_past[:, -1]
                )
            else:
                future_pos = X / np.linalg.norm(X) * RADIUS + X_past[:, -1].numpy()

        else:
            future_perp /= np.linalg.norm(future_perp)
            present_perp /= np.linalg.norm(present_perp)
            t = X @ present_perp
            offset = t * future_perp
            offset = offset / np.linalg.norm(offset) * RADIUS
            offset = future_perp  # future pos is the orthogonal vector of the 2 future positions
            future_pos = X_future[:2, -1] + offset

        #######################  Generating the input for the model  #############################
        heading = generated_traj[:, 1:] - generated_traj[:, :-1]

        # generating noise for the robot's velocities
        sigma_v = 0.15
        N_v = np.random.rand(*heading.shape) * sigma_v
        heading = torch.tensor(heading)
        heading += torch.tensor(N_v, dtype=torch.float)

        # shifting vectors into the robot's coordinate frame
        relative_vectors_r, robot_velocities_r, future_pos_r = (
            ConvertAbsoluteToRobotFrame(X_past, generated_traj, heading, future_pos)
        )

        # creating the input array to the model
        input_vectors_r = ProcessPast(relative_vectors_r, robot_velocities_r)

        # plt.show()

        return (
            input_vectors_r,
            future_pos_r,
            X_past,
            X_future,
            generated_traj,
        )


##################  Dataset for the non-generated trajectories  #################
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


################  Old Debugging Plotting Methods  ###################
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


def GraphTrajectory(X_past, X_future, input_vectors, generated_traj, future_pos):
    ##########  Graphing generated trajectory ###########
    fig, ax = plt.subplots(figsize=(9, 9))
    # ax.set_xlim(-5, 5)
    # ax.set_ylim(-5, 5)
    ax.scatter(
        X_past[0] - generated_traj[0, -1],
        X_past[1] - generated_traj[1, -1],
        label="Original Past Trajectory",
    )
    ax.scatter(
        X_future[0] - generated_traj[0, -1],
        X_future[1] - generated_traj[1, -1],
        label="Future Trajectory",
    )
    ax.scatter(
        generated_traj[0] - generated_traj[0, -1],
        generated_traj[1] - generated_traj[1, -1],
        label="Generated Robot Trajectory",
    )

    for i in range(len(input_vectors[0])):
        ax.arrow(
            generated_traj[0][i] - generated_traj[0][-1],
            generated_traj[1][i] - generated_traj[1][-1],
            input_vectors[0][i],
            input_vectors[1][i],
            color="black",
            width=0.001,
            head_width=0.001 * 5,
        )

    ax.scatter(
        future_pos[0],
        future_pos[1],
        color="red",
        label="Future Position",
    )
    ax.set_aspect("equal")
    ax.legend()
    fig.suptitle("Trajectories")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()
