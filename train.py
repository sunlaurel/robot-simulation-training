from utils import *
from train_helper import *

# TODO: make sure that the predicted velocities are converted to float tensors instead of double tensors

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
scale_flag = True

# Splitting the data 80/20, just cutting at 80% of the data
csv_file = "./training-data/crowd_data.csv"
csv_data = pd.read_csv(csv_file)
split = int(0.8 * len(csv_data))

# Initalizing the dataset
training_data = TrajectoryDataset(
    csv_data=csv_data[:split],
    offset_flag=offset_flag,
    rotate_flag=rotate_flag,
    noise_flag=noise_flag,
    scale_flag=scale_flag,
)

testing_data = TrajectoryDataset(
    csv_data=csv_data[split:]
)

# Set optimizer (adam) and loss function (mse)
network = models.MultiLayer(4 * past_steps, 100, 100, future_steps * 4)
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
loss_function = nn.L1Loss()


if __name__ == "__main__":
    # Load the data, and split it into batches
    training_generator = torch.utils.data.DataLoader(
        training_data, batch_size=batch_size, shuffle=True
    )

    testing_generator = torch.utils.data.DataLoader(
        testing_data, batch_size=batch_size, shuffle=False
    )

    print("Loaded Data")

    # Visualizing the path
    # for x_past, x_future, v_past, v_future in training_generator:
    #     # Plot the trajectory
    #     plot_trajectory(x_past[0], x_future[0], v_past[0], v_future[0])
    #     break  # Only plot the first batch to avoid unnecessary looping

    trainAndGraph(
        network,
        training_generator,
        testing_generator,
        loss_function,
        optimizer,
        num_epochs,
        print_interval,
    )
