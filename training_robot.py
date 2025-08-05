from utils import *
from train_robot_helper import *

# TODO: reorganize so that all the training scripts are in one folder
# TODO: reorganize so that all the simulation scripts are in another folder

# loading in config.json
with open(file="./utils/config.json", mode="r") as file:
    data = json.load(file)

# Training parameters
num_epochs = 150
print_interval = 1
learning_rate = 0.001
batch_size = 100
past_steps = data["past-steps"]
future_steps = data["future-steps"]
# num_features = data["num-features"]  # num_features indicates the number of variables (ie: x-pos, y-pos, x-velocity, y-velocity)

# Setting the flags
offset = data["offset"]
rotate = data["rotate"]
add_noise = data["add-noise"]
scale = data["scale"]

# Set optimizer (adam) and loss function (mae)
# breakpoint()
network = MultiLayerRobot(4 * past_steps, 100, 100, 2)
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
loss_function = nn.L1Loss()


if __name__ == "__main__":
    # Load the data, and split it into batches
    training_data, testing_data = GenTrainTestGeneratedDatasets(
        csv_path="./data/training-data/crowd_data.csv",
        past_steps=past_steps,
        future_steps=future_steps,
    )

    training_generator = torch.utils.data.DataLoader(
        training_data, batch_size=batch_size, shuffle=True, num_workers=6
    )

    testing_generator = torch.utils.data.DataLoader(
        testing_data, batch_size=batch_size, shuffle=False, num_workers=6
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
        future_steps,
        past_steps,
        print_interval,
        offset=offset,
        add_noise=add_noise,
        scale=scale,
        rotate=rotate,
    )
