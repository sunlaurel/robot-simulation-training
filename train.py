import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torch.nn.functional as F
from utils import *
from train_helper import *

# Training parameters
num_epochs = 100
print_interval = 1
learning_rate = 0.001
batch_size = 100
past_steps = 5
future_steps = 9

# Setting the flags
offset_flag = True
rotate_flag = False
noise_flag = False
scale_flag = False

# Iniitalizing the dataset
dataset = data.TrajectoryDataset(
    "./training-data/crowd_data.csv",
    offset_flag=offset_flag,
    rotate_flag=rotate_flag,
    noise_flag=noise_flag,
    scale_flag=scale_flag,
)

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

for x_past, x_future, v_past, v_future in training_generator:
    # Plot the trajectory
    plot_trajectory(x_past[0], x_future[0], v_past[0], v_future[0])
    # break  # Only plot the first batch to avoid unnecessary looping

# trainAndGraph(
#     network,
#     training_generator,
#     testing_generator,
#     loss_function,
#     optimizer,
#     num_epochs,
#     print_interval,
# )

# _, batch = next(enumerate(training_generator))
# plt.plot(batch[0][0][0], batch[0][0][1], 'ro')
# plt.plot(batch[1][0][0], batch[1][0][1], 'bo')

# with torch.no_grad():
#   predicted = network(batch[0])

# print("past data: ", batch[0][0])
# print("predicted path: ", predicted[0])
# print("actual path: ", batch[1][0])
# # plt.gca().set_aspect('equal')
# plt.plot(predicted[0][0], predicted[0][1], 'go', mfc="none")

# # plt.xlim(-2, 2)
# # plt.ylim(-2, 2)
