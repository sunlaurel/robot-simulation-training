import utils
import torch
from torch.utils.data import DataLoader
import json
import matplotlib.pyplot as plt
from train_helper import T_test


def plot_predicted_trajectory(x_past, x_future, x_predicted):
    """Graphs the predicted trajectory compared to the actual trajectory"""
    # fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    x_past_coords = x_past[0, :].numpy()
    y_past_coords = x_past[1, :].numpy()
    x_future_coords = x_future[0, :].numpy()
    y_future_coords = x_future[1, :].numpy()
    x_predicted_coords = x_predicted[0, :].numpy()
    y_predicted_coords = x_predicted[1, :].numpy()

    # graphing the maintaining velocity model
    x_baseline = utils.models.constant_velocity_model(x_past)
    x_baseline_coords = x_baseline[0, :].numpy()
    y_baseline_coords = x_baseline[1, :].numpy()

    # graphing the stand still model
    x_stand = utils.models.stand_model(x_past)
    x_stand_coords = x_stand[0, :].numpy()
    y_stand_coords = x_stand[1, :].numpy()

    plt.plot(x_past_coords, y_past_coords, marker="o", linestyle="-", color="r")
    plt.plot(x_future_coords, y_future_coords, marker="o", linestyle="-", color="b")
    plt.plot(
        x_predicted_coords, y_predicted_coords, marker="o", linestyle="-", color="g"
    )
    plt.plot(
        x_baseline_coords,
        y_baseline_coords,
        marker="o",
        linestyle="-",
        color=(0.6, 0.3, 0.9),
    )
    plt.plot(
        x_stand_coords, y_stand_coords, marker="o", linestyle="-", color=(1, 0.647, 0)
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Positions")
    plt.legend(
        [
            "Past trajectory",
            "Future trajectory",
            "Predicted trajectory",
            "Maintain velocity baseline trajectory",
            "Standing still baseline trajectory",
        ]
    )
    plt.show()


if __name__ == "__main__":

    with open(file="./utils/config.json", mode="r", encoding="utf-8") as file:
        data = json.load(file)

    # Setting the constants
    offset_flag = data["offset"]
    scale_flag = data["scale"]
    noise_flag = data["add-noise"]
    rotate_flag = data["rotate"]
    past_steps = data["past-steps"]
    future_steps = data["future-steps"]

    network = utils.models.MultiLayer(
        input_size=2 * past_steps,
        hidden_layer1=100,
        hidden_layer2=100,
        output_size=2 * future_steps,
    )

    save_path = f"./weights/best-weights/best_weight{'_noise' if noise_flag else ''}{'_rotate' if rotate_flag else ''}{'_scale' if scale_flag else ''}{'_offset' if offset_flag else ''}{'(' + str(past_steps) + '-past)' if past_steps != 10 else ''}{'(0.1-sigma)' if noise_flag else ''}.pth"
    print("Model visualized:", save_path)
    network.load_state_dict(torch.load(save_path, weights_only=True))

    _, testing_data = utils.data.GenTrainTestDatasets(
        "./data/training-data/crowd_data.csv", past_steps, future_steps
    )

    data_loader = DataLoader(testing_data, batch_size=1, shuffle=True)

    for x_past, x_future in data_loader:
        x_past, x_future = T_test(
            x_past, x_future, offset=offset_flag, scale=scale_flag
        )

        # Plot the trajectory
        with torch.no_grad():
            # predicted = network(
            #     torch.cat((x_past.float(), v_past.float()), dim=1)
            # )
            predicted = network(x_past.float()).squeeze()

        plot_predicted_trajectory(
            x_past=x_past[0],
            x_future=x_future[0],
            # v_past=v_past[0],
            # v_future=v_future[0],
            x_predicted=predicted[0:2],
            # v_predicted=predicted[2:],
        )

        # break  # Only plot the first batch to avoid unnecessary looping
