import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torch.nn.functional as F
from utils import *
from train_helper import *

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

# Training parameters
num_epochs = 100
print_interval = 1
learning_rate = 0.001
batch_size = 100
past_steps = 5
future_steps = 9

# Iniitalizing the dataset
dataset = data.TrajectoryDataset("./training-data/crowd_data.csv")

# Set optimizer (adam) and loss function (mse)
network = models.MultiLayer(4 * past_steps, 100, 100, future_steps * 4)
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
loss_function = nn.L1Loss()

# Load the data, and split it into batches
training_generator = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True
)

testing_generator = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=False
)

print("Loaded Data")

trainAndGraph(
    network,
    training_generator,
    testing_generator,
    loss_function,
    optimizer,
    num_epochs,
    learning_rate,
    print_interval,
)

for x_past, x_future, v_past, v_future in training_generator:
    # Plot the trajectory
    plot_trajectory(x_past[0], x_future[0], v_past[0], v_future[0])
    # break  # Only plot the first batch to avoid unnecessary looping