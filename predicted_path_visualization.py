from utils import *
from training import testing_data, network, offset, add_noise, rotate, scale
from train_helper import T_test
from baseline_models import baseline_model, stand_model


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
    x_baseline = baseline_model(x_past)
    x_baseline_coords = x_baseline[0, :].numpy()
    y_baseline_coords = x_baseline[1, :].numpy()
    
    # graphing the stand still model
    x_stand = stand_model(x_past)
    x_stand_coords = x_stand[0, :].numpy()
    y_stand_coords = x_stand[1, :].numpy()
    

    plt.plot(x_past_coords, y_past_coords, marker="o", linestyle="-", color="r")
    plt.plot(x_future_coords, y_future_coords, marker="o", linestyle="-", color="b")
    plt.plot(
        x_predicted_coords, y_predicted_coords, marker="o", linestyle="-", color="g"
    )
    plt.plot(
        x_baseline_coords, y_baseline_coords, marker="o", linestyle="-", color=(0.6, 0.3, 0.9)
    )
    plt.plot(
        x_stand_coords, y_stand_coords, marker="o", linestyle="-", color=(1, 0.647, 0)
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Positions")
    plt.show()

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


if __name__ == "__main__":
    
    # save_path = f"./best-weights/best_weight{'_noise' if add_noise else ''}{'_rotate' if rotate else ''}{'_scale' if scale else ''}{'_offset' if offset else ''}.pth"
    save_path = "./best-weights/best_weight_offset.pth"
    print("model visualized:", save_path)
    network.load_state_dict(torch.load(save_path))

    data_loader = DataLoader(testing_data, batch_size=1, shuffle=True)

    for x_past, x_future in data_loader:
        x_past, x_future = T_test(x_past, x_future, offset=offset, scale=scale)

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
