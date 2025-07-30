import random
import pandas as pd
import ast
import torch
import numpy as np
import matplotlib.pyplot as plt
import utils

df = pd.read_csv("./simulation-data/sim1.csv")

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
    plt.figure(figsize=(12, 12))
    plt.scatter(agent_pos[0, len(agent_pos[0]) - 10:], agent_pos[1, len(agent_pos[0]) - 10:], label="Agent")
    plt.scatter(robot_pos[0], robot_pos[1], label="Robot")
    plt.scatter(target_pos[0], target_pos[1], label="Predicted Target")
    plt.scatter(
        new_predicted[0] + agent_pos[0, -1],
        new_predicted[1] + agent_pos[1, -1],
        label="New Predicted",
    )
    # plt.quiver(
    #     [agent_pos[0, -1], robot_pos[0, -1]],
    #     [agent_pos[1, -1], robot_pos[1, -1]],
    #     [agent_pos[0, -1] - agent_pos[0, -2], robot_pos[0, -1] - robot_pos[0, -2]],
    #     [agent_pos[1, -1] - agent_pos[1, -2], robot_pos[1, -1] - robot_pos[1, -2]],
    # )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Simulation Visualization")
    print("predicted == new predicted:", new_predicted + agent_pos[:, -1] == target_pos)
    # plt.xlim(0, 10)
    # plt.ylim(0, 8)

    plt.show()
