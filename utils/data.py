from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split


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

        # generating random slope from -2 to 2
        slope = random.random() * 4 - 1
        # some initial random offset from -1 to 1
        offset = random.random() * 2 - 1
        # step is evenly spaced by the end and start x coords
        step = (X_past[0, -1] - X_past[0, 0]) / self.N_past

        # starting points for the generated line
        generated_x = X_past[0, 0]
        generated_y = X_past[1, 0] + offset

        generated_traj = [[generated_x], [generated_y]]
        for j in range(self.N_past - 1):
            generated_traj[0].append(generated_x + step)
            generated_traj[1].append(generated_y + slope * step)
            generated_x += step
            generated_y += slope * step

        past_relative_vectors = X_past - generated_traj

        # future point is the 1m from the person's right side
        future_pos = X_future[:, -1] - X_future[:, -2]
        future_pos = torch.tensor(
            [-(X_future[1, -1] - X_future[1, -2]), X_future[0, -1] - X_future[0, -2]]
        )
        future_pos /= np.linalg.norm(future_pos)
        breakpoint()
        future_vector = (np.add(X_future[:, -1], future_pos)) - np.array()
        plt.scatter(
            X_past[0],
            X_past[1],
            label="Original Trajectory",
        )

        plt.scatter(X_future[0], X_future[1], label="Future Trajectory")
        plt.scatter(generated_traj[0], generated_traj[1], label="Generated Trajectory")
        plt.quiver(
            X_future[0, -1],
            X_future[1, -1],
            future_pos[0],
            future_pos[1],
            label="Offset from future trajectory",
            color="purple",
        )
        plt.scatter(
            X_future[0, -1] + future_pos[0],
            X_future[1, -1] + future_pos[1],
            color="red",
            label="Future position",
        )
        plt.quiver(
            generated_traj[0][-1],
            generated_traj[1][-1],
            future_vector[0],
            future_vector[1],
            label="Robot Future Vector",
            color="gray",
        )
        plt.legend()
        plt.show()

        return past_relative_vectors, future_vector


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
        # V_past = V[:, random_frame - self.N_past + 1 : random_frame + 1]
        # V_future = V[:, random_frame + 1 : random_frame + 1 + self.N_future]

        return X_past, X_future  # , V_past, V_future

        # # Convert past and future points to tensors
        # past_x = torch.tensor(past_points.iloc[:, 0].values, dtype=torch.float32)
        # past_y = torch.tensor(past_points.iloc[:, 1].values, dtype=torch.float32)
        # future_x = torch.tensor(future_points.iloc[:, 0].values, dtype=torch.float32)
        # future_y = torch.tensor(future_points.iloc[:, 1].values, dtype=torch.float32)

        # # Concatenate past and future to form a trajectory
        # x_trajectory = torch.cat([past_x, future_x])
        # y_trajectory = torch.cat([past_y, future_y])
        # return x_trajectory, y_trajectory


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


# # Example usage:
# csv_file = "entire_scene_crowd_data_sample.csv"  # Path to your CSV file
# dataset = TrajectoryDataset(csv_file)
# # dataset.__getitem__(300)


# # Create a DataLoader
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# # Iterate through the DataLoader
# for x_past, x_future, v_past, v_future in dataloader:
#     # Plot the trajectory
#     plot_trajectory(x_past[0], x_future[0], v_past[0], v_future[0])
#     # break  # Only plot the first batch to avoid unnecessary looping


""" For predicted path visualization and graphing it out in case I need it again """
# dt = 0.12
# vx_past_coords = v_past[0, :].numpy()
# vy_past_coords = v_past[1, :].numpy()
# vx_future_coords = v_future[0, :].numpy()
# vy_future_coords = v_future[1, :].numpy()
# vx_predicted_coords = v_predicted[0, :].numpy()
# vy_predicted_coords = v_predicted[1, :].numpy()

# axes[0].plot(x_past_coords, y_past_coords, marker="o", linestyle="-", color="r")
# axes[0].plot(x_future_coords, y_future_coords, marker="o", linestyle="-", color="b")
# axes[0].plot(
#     x_predicted_coords, y_predicted_coords, marker="o", linestyle="-", color="g"
# )
# axes[0].set_xlabel("x")
# axes[0].set_ylabel("y")
# axes[0].set_title("Positions")

# axes[1].plot(x_past_coords, y_past_coords, marker="o", linestyle="-", color="k")
# axes[1].plot(x_future_coords, y_future_coords, marker="o", linestyle="-", color="k")
# axes[1].quiver(
#     x_past_coords,
#     y_past_coords,
#     vx_past_coords,
#     vy_past_coords,
#     color="r",
#     angles="xy",
#     scale_units="xy",
#     scale=1 / dt,
# )
# axes[1].quiver(
#     x_future_coords,
#     y_future_coords,
#     vx_future_coords,
#     vy_future_coords,
#     color="b",
#     angles="xy",
#     scale_units="xy",
#     scale=1 / dt,
# )
# axes[1].quiver(
#     x_predicted_coords,
#     y_predicted_coords,
#     vx_predicted_coords,
#     vy_predicted_coords,
#     color="g",
#     angles="xy",
#     scale_units="xy",
#     scale=1 / dt,
# )

# axes[1].set_title("Velocity Vectors")
# axes[1].set_xlabel("x")
# axes[1].set_ylabel("y")
# axes[1].grid(True)

# fig.tight_layout()
# plt.show()
