import utils
import torch
from torch.utils.data import DataLoader
import json
import matplotlib.pyplot as plt
from train_helper import T_test


def plot_predicted_trajectory(x_past, x_future, x_robot, x_target, x_predicted):
    """Graphs the predicted trajectory compared to the actual trajectory"""
    # breakpoint()
    x_past_coords = x_past[0].numpy()
    y_past_coords = x_past[1].numpy()
    x_future_coords = x_future[0].numpy()
    y_future_coords = x_future[1].numpy()
    x_robot_traj_coords = x_robot[0].numpy()
    y_robot_traj_coords = x_robot[1].numpy()
    x_target_coord = x_target[0] + x_robot[0, -1]
    y_target_coord = x_target[1] + x_robot[1, -1]
    x_predicted_coord = x_predicted[0] + x_robot[0, -1]
    y_predicted_coord = x_predicted[1] + x_robot[1, -1]

    plt.figure(figsize=(9, 9))
    plt.scatter(x_past_coords, y_past_coords, marker="o", color="r")
    plt.scatter(x_future_coords, y_future_coords, marker="o", color="purple")
    plt.plot(
        x_robot_traj_coords,
        y_robot_traj_coords,
        marker="o",
        linestyle="-",
        color="orange",
    )
    plt.scatter(x_target_coord, y_target_coord, marker="o", color="b")
    plt.scatter(x_predicted_coord, y_predicted_coord, marker="o", color="g")
    # plt.xlim(-5, 5)
    # plt.ylim(-5, 5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Trajectories")
    plt.legend(
        [
            "Past Trajectory",
            "Future Trajectory",
            "Generated Robot Trajectory",
            "Actual Future Robot Position",
            "Predicted Future Robot Position",
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

    network = utils.models.MultiLayerRobot(
        input_size=4 * past_steps,
        hidden_layer1=100,
        hidden_layer2=100,
        output_size=2,
    )

    save_path = f"./best-weights-robot/best_weight{'_noise' if noise_flag else ''}{'_rotate' if rotate_flag else ''}{'_scale' if scale_flag else ''}{'_offset' if offset_flag else ''}{'(' + str(past_steps) + '-past)' if past_steps != 10 else ''}{'(0.1-sigma)' if noise_flag else ''}.pth"
    print("Model visualized:", save_path)
    network.load_state_dict(torch.load(save_path, weights_only=True))

    training_data, testing_data = utils.data.GenTrainTestGeneratedDatasets(
        "./training-data/crowd_data.csv", past_steps, future_steps
    )

    data_loader = DataLoader(training_data, batch_size=1, shuffle=True)

    for relative_past, target_pos, X_past, X_future, X_robot in data_loader:
        relative_past, target_pos = T_test(
            relative_past, target_pos, offset=offset_flag, scale=scale_flag
        )

        # Plot the trajectory
        with torch.no_grad():
            predicted = network(relative_past.float()).squeeze()

        if scale_flag:
            target_pos /= 0.5
            predicted /= 0.5

        predicted_w = utils.data.ConvertRobotFrameToAbsolute(X_robot[0], predicted)
        target_pos_w = utils.data.ConvertRobotFrameToAbsolute(X_robot[0], target_pos[0])

        predicted_w += X_robot[0, :, -1]
        target_pos_w += X_robot[0, :, -1]

        plot_predicted_trajectory(
            x_past=X_past[0],
            x_future=X_future[0],
            x_robot=X_robot[0],
            x_target=target_pos_w,
            x_predicted=predicted_w,
        )

        # break  # Only plot the first batch to avoid unnecessary looping
