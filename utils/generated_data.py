from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import random
import torch
import numpy as np
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
        step = (
            X_past[0, random_frame + 1] - X_past[random_frame - self.N_past + 1, 0]
        ) / self.N_past

        # starting points for the generated line
        generated_x = X_past[0, random_frame - self.N_past + 1]
        generated_y = X_past[1, random_frame - self.N_past + 1] + offset

        generated_traj = [[generated_x], [generated_y]]
        for j in range(self.N_past - 1):
            generated_traj[0].append(generated_x + step)
            generated_traj[1].append(generated_y + slope * step)
            generated_x += step
            generated_y += slope * step

        plt.scatter(
            X_past[0, random_frame - self.N_past + 1],
            X_past[1, random_frame - self.N_past + 1],
            label="Original Trajectory",
        )
        plt.scatter(generated_traj[0], generated_traj[1], label="Generated Trajectory")
        # breakpoint()
        plt.legend(title="Trajectories Key", loc="upper right")
        plt.title("Trajectory Visualization")
        plt.xlabel("x position")
        plt.ylabel("y position")
        plt.show()

        past_relative_vectors = X_past - generated_traj

        # future point is the 1m from the person's right side
        future_vector = X_future[:, -1] - X_future[:, -2]
        future_vector = torch.tensor(
            -(X_future[1, -1] - X_future[1, -2]), X_future[0, -1] - X_future[0, -2]
        )
        future_vector /= np.linalg.norm(future_vector)

        future_vector = future_vector - X_past[:, -1]

        return past_relative_vectors, future_vector
