from utils import *
from train_helper import *


# TODO: NEED TO FIGURE OUT HOW TO REMOVE CIRCULAR IMPORTS
# TODO: ORGANIZE IMPORT ORGANIZATION


# Training parameters
num_epochs = 700
print_interval = 1
learning_rate = 0.001
batch_size = 100
past_steps = 10
future_steps = 10

# Setting the flags
offset = True
rotate = False
add_noise = False
scale = False

# Splitting the data 80/20, just cutting at 80% of the data
training_data, testing_data = GenTrainTestDatasets(
    "./training-data/crowd_data.csv", past_steps=past_steps, future_steps=future_steps
)

# Set optimizer (adam) and loss function (mse)
# breakpoint()
network = MultiLayer(2 * past_steps, 100, 100, 2 * future_steps)
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
        offset=offset,
        add_noise=add_noise,
        scale=scale,
        rotate=rotate,
    )
