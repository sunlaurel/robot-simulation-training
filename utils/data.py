from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import torch
import numpy as np
import random
from sklearn.model_selection import train_test_split

def GenTrainTestDatasets(csv_path):
        csv_data = pd.read_csv(csv_path)
        person_ids = csv_data["id"].unique()
        train_ids, test_ids = train_test_split(person_ids, test_size=0.2, random_state=2244)
        train_dataset = TrajectoryDataset(csv_data, train_ids)
        test_dataset = TrajectoryDataset(csv_data, test_ids)
        return train_dataset, test_dataset
    
class TrajectoryDataset(Dataset):

    def __init__(self, csv_data, split_ids):
        self.N_past = 5
        self.N_future = 9
        self.person_ids = csv_data["id"].unique()
        self.position_data = {}  # mapping from id to trajectory
        self.velocity_data = {}  # mapping from id to trajectory
        self.len = len(csv_data)

        self.person_ids = split_ids        
        self.position_data = {} # mapping from id to trajectory
        # self.velocity_data = {} # mapping from id to trajectory
        for person_id in tqdm(self.person_ids, total = len(self.person_ids), dynamic_ncols=True):
            trajdata = csv_data[csv_data["id"] == person_id].sort_values(by="time")
            if len(trajdata) < self.N_past + self.N_future:
                continue
            self.position_data[person_id] = trajdata[["x","y"]].to_numpy().T # 2 x N
            self.velocity_data[person_id] = trajdata[["v_x","v_y"]].to_numpy().T # 2 x N

        self.person_ids = list(self.position_data.keys()) # remove people with short trajectories


    def __len__(self):
        # The dataset length is the number of rows in the CSV file
        return self.len

    def __getitem__(self, idx):
        random_person_id = int(random.choice(self.person_ids))
        X = self.position_data[random_person_id]
        V = self.velocity_data[random_person_id]
        random_frame = int(random.randint(self.N_past, X.shape[1] - 1 - self.N_future))

        # Determine the indices for past and future points
        X_past = X[:, random_frame - self.N_past + 1 : random_frame + 1]
        X_future = X[:, random_frame + 1 : random_frame + 1 + self.N_future]
        V_past = V[:, random_frame - self.N_past + 1 : random_frame + 1]
        V_future = V[:, random_frame + 1 : random_frame + 1 + self.N_future]
        
        return X_past, X_future#, V_past, V_future

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
