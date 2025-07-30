import random
import pandas as pd
import ast
import torch
import numpy as np
import matplotlib.pyplot as plt
import utils

df = pd.read_csv("./simulation-data/sim2.csv")

agent_pos_lst = df["agent_past_pos"]
robot_pos_lst = df["robot_past_pos"]
target_pos_lst = df["predicted_target_pos"]

network = utils.models.MultiLayerRobot(
    input_size=4 * 10,
    hidden_layer1=100,
    hidden_layer2=100,
    output_size=2,
)

save_path = "./best-weights-robot/best_weight.pth"
print("Model visualized:", save_path)
network.load_state_dict(torch.load(save_path, weights_only=True))


while True:
    i = random.randint(10, len(df) - 1)
    agent_pos = np.array(ast.literal_eval(agent_pos_lst[i]))
    robot_pos = np.array(ast.literal_eval(robot_pos_lst[i]))
    target_pos = ast.literal_eval(target_pos_lst[i])

    input_vectors = utils.data.ProcessPast(agent_pos, robot_pos)

    with torch.no_grad():
        new_predicted = network(torch.tensor(input_vectors).float().unsqueeze(0))

    # breakpoint()
    new_predicted = new_predicted.squeeze()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        agent_pos[0, len(agent_pos[0]) - 10 :],
        agent_pos[1, len(agent_pos[0]) - 10 :],
        label="Agent",
    )
    ax.scatter(robot_pos[0], robot_pos[1], label="Robot")
    ax.scatter(target_pos[0], target_pos[1], label="Predicted Target")
    # ax.scatter(
    #     new_predicted[0] + agent_pos[0, -1],
    #     new_predicted[1] + agent_pos[1, -1],
    #     label="New Predicted",
    # )
    for i in range(1, len(robot_pos[0])):
        ax.arrow(
            robot_pos[0, i - 1],
            robot_pos[1, i - 1],
            robot_pos[0, i] - robot_pos[0, i - 1],
            robot_pos[1, i] - robot_pos[1, i - 1],
            color="black",
            width=0.003,
            head_width=6 * 0.003,
        )
        ax.arrow(
            agent_pos[0, i - 1],
            agent_pos[1, i - 1],
            agent_pos[0, i] - agent_pos[0, i - 1],
            agent_pos[1, i] - agent_pos[1, i - 1],
            color="black",
            width=0.003,
            head_width=6 * 0.003,
        )

    plt.gca().set_aspect("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Simulation Visualization")
    # plt.xlim(0, 10)
    # plt.ylim(0, 8)

    plt.show()
